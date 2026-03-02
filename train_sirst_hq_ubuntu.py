import os
import time
import argparse
import json
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
try:
    from skimage import measure
except Exception:
    measure = None
# Cross-version AMP import (PyTorch>=2.0 uses torch.amp, older uses torch.cuda.amp)
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new  # type: ignore
    def autocast_ctx(device: str):
        if device.startswith("cuda") and torch.cuda.is_available():
            return _autocast_new("cuda")
        return nullcontext()
    def make_scaler(device: str):
        if device == "cuda" and torch.cuda.is_available():
            return _GradScaler_new("cuda")
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        return _DummyScaler()
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old  # type: ignore
    def autocast_ctx(device: str):
        if device.startswith("cuda") and torch.cuda.is_available():
            return _autocast_old()
        return nullcontext()
    def make_scaler(device: str):
        if torch.cuda.is_available():
            return _GradScaler_old()
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        return _DummyScaler()


from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_backbone_bifusion_block_adapter,
    build_bifusion_adapter_lite,
    build_gated_backbone_bifusion_block_adapter,
    build_text_conditioner,
    build_text_dense_mask_prompt_generator,
    build_text_dense_mask_prompt_generator_v2,
    build_text_sparse_prompt_projector,
)


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


class NWDLoss(nn.Module):
    def __init__(self, constant: float = 12.0, eps: float = 1e-6):
        super().__init__()
        self.C = float(constant)
        self.eps = float(eps)

    def get_gaussian_params(self, mask, grid_x, grid_y):
        mass = mask.sum(dim=(2, 3), keepdim=True) + self.eps
        mu_x = (mask * grid_x).sum(dim=(2, 3), keepdim=True) / mass
        mu_y = (mask * grid_y).sum(dim=(2, 3), keepdim=True) / mass
        var_x = (mask * (grid_x - mu_x).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
        var_y = (mask * (grid_y - mu_y).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
        sigma_x = torch.sqrt(var_x + self.eps)
        sigma_y = torch.sqrt(var_y + self.eps)
        return mu_x, mu_y, sigma_x, sigma_y

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)
        B, _, H, W = probs.shape
        device = probs.device
        y = torch.arange(H, device=device, dtype=probs.dtype) + 0.5
        x = torch.arange(W, device=device, dtype=probs.dtype) + 0.5
        try:
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(y, x)
        valid_mask = (targets.sum(dim=(2, 3)) > 0).float().view(B)

        mu_x_p, mu_y_p, sig_x_p, sig_y_p = self.get_gaussian_params(probs, grid_x, grid_y)
        mu_x_t, mu_y_t, sig_x_t, sig_y_t = self.get_gaussian_params(targets, grid_x, grid_y)

        wd2 = (mu_x_p - mu_x_t).pow(2) + (mu_y_p - mu_y_t).pow(2) + \
              (sig_x_p - sig_x_t).pow(2) + (sig_y_p - sig_y_t).pow(2)
        wd2 = wd2.view(B)
        nwd = torch.exp(-torch.sqrt(wd2 + self.eps) / self.C)
        loss = 1.0 - nwd
        return (loss * valid_mask).sum() / (valid_mask.sum() + self.eps)


def radial_frequency_profile(mask: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Compute normalized radial energy profile for each mask."""
    if mask.dim() != 4:
        raise ValueError("mask tensor must have shape [B, C, H, W]")
    B, C, H, W = mask.shape
    mask = mask.float()
    spec = torch.fft.rfft2(mask, dim=(-2, -1), norm="forward")
    energy = spec.real.pow(2) + spec.imag.pow(2)

    fy = torch.fft.fftfreq(H, d=1.0, device=mask.device)
    fx = torch.fft.rfftfreq(W, d=1.0, device=mask.device)
    fy = fy.to(mask.dtype).view(1, 1, H, 1)
    fx = fx.to(mask.dtype).view(1, 1, 1, fx.numel())
    radius = torch.sqrt(fy.pow(2) + fx.pow(2))
    radius = radius / radius.max().clamp(min=1e-6)
    bin_idx = torch.clamp((radius * (num_bins - 1)).long(), max=num_bins - 1)

    energy_flat = energy.reshape(B, C, -1)
    idx_flat = bin_idx.reshape(1, 1, -1).expand_as(energy_flat)
    profile = torch.zeros(B, C, num_bins, device=mask.device, dtype=energy.dtype)
    profile.scatter_add_(2, idx_flat, energy_flat)

    counts = torch.zeros(num_bins, device=mask.device, dtype=energy.dtype)
    counts.scatter_add_(0, bin_idx.reshape(-1), torch.ones(bin_idx.numel(), device=mask.device, dtype=energy.dtype))
    counts = counts.clamp_min_(1.0)
    profile = profile / counts.view(1, 1, -1)
    profile = profile / (profile.sum(dim=-1, keepdim=True) + 1e-6)
    return profile


class PD_FA:
    def __init__(self, distance_thresh: int = 3):
        if measure is None:
            raise RuntimeError("scikit-image is required for PD/FA metrics; please install scikit-image.")
        self.distance_thresh = int(distance_thresh)
        self.reset()

    def update(self, preds, labels, size_hw):
        predits = np.array(preds.cpu()).astype("int64")
        labelss = np.array(labels.cpu()).astype("int64")

        image = measure.label(predits, connectivity=2)
        coord_image = list(measure.regionprops(image))
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.all_pixel += int(size_hw[0] * size_hw[1])

        matched = 0
        for region in coord_label:
            centroid_label = np.array(region.centroid)
            for idx in range(len(coord_image)):
                centroid_image = np.array(coord_image[idx].centroid)
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < self.distance_thresh:
                    matched += 1
                    del coord_image[idx]
                    break

        unmatched_areas = [r.area for r in coord_image]
        self.dismatch_pixel += int(np.sum(unmatched_areas)) if unmatched_areas else 0
        self.PD += matched

    def get(self):
        pd = self.PD / (self.target + 1e-6)
        fa = self.dismatch_pixel / (self.all_pixel + 1e-6)
        return float(pd), float(fa)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


def _boundary_map_from_mask(mask_2d_float: torch.Tensor) -> torch.Tensor:
    # mask_2d_float: [H,W] in {0,1}
    m = mask_2d_float.unsqueeze(0).unsqueeze(0)
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    erode = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    b = (dil - erode).clamp(min=0.0, max=1.0)
    return (b[0, 0] > 0).to(mask_2d_float.dtype)


def sample_points_from_mask(mask_bhw: torch.Tensor, n_pos=4, n_neg=4, boundary_prior: bool = False, boundary_ratio: float = 0.5):
    B, H, W = mask_bhw.shape
    device = mask_bhw.device
    pts, labels = [], []
    for b in range(B):
        pos_idx = (mask_bhw[b] > 0).nonzero(as_tuple=False)
        neg_idx = (mask_bhw[b] == 0).nonzero(as_tuple=False)
        if boundary_prior:
            bmap = _boundary_map_from_mask(mask_bhw[b].float())
            bpos = ((mask_bhw[b] > 0) & (bmap > 0)).nonzero(as_tuple=False)
            bneg = ((mask_bhw[b] == 0) & (bmap > 0)).nonzero(as_tuple=False)
            # how many from boundary
            bp = int(min(n_pos, len(pos_idx)) * boundary_ratio)
            bn = int(min(n_neg, len(neg_idx)) * boundary_ratio)
            sel_bpos = bpos[torch.randint(len(bpos), (bp,), device=bpos.device)] if bp > 0 and len(bpos) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            sel_bneg = bneg[torch.randint(len(bneg), (bn,), device=bneg.device)] if bn > 0 and len(bneg) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            rem_p = max(0, min(n_pos, len(pos_idx)) - sel_bpos.size(0))
            rem_n = max(0, min(n_neg, len(neg_idx)) - sel_bneg.size(0))
            sel_pos_rest = pos_idx[torch.randint(len(pos_idx), (rem_p,), device=pos_idx.device)] if rem_p > 0 and len(pos_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            sel_neg_rest = neg_idx[torch.randint(len(neg_idx), (rem_n,), device=neg_idx.device)] if rem_n > 0 and len(neg_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            pos = torch.cat([sel_bpos, sel_pos_rest], dim=0)
            neg = torch.cat([sel_bneg, sel_neg_rest], dim=0)
        else:
            # original purely random sampling
            pass
        npos = min(n_pos, len(pos_idx)) if len(pos_idx) > 0 else 0
        nneg = min(n_neg, len(neg_idx)) if len(neg_idx) > 0 else 0
        if not boundary_prior:
            pos = (pos_idx[torch.randint(len(pos_idx), (npos,), device=pos_idx.device)] if npos > 0 else torch.zeros((0, 2), dtype=torch.long, device=device))
            neg = (neg_idx[torch.randint(len(neg_idx), (nneg,), device=neg_idx.device)] if nneg > 0 else torch.zeros((0, 2), dtype=torch.long, device=device))
        p = torch.cat([pos, neg], dim=0)
        l = torch.cat([torch.ones(npos), torch.zeros(nneg)], dim=0)
        if p.numel() == 0:
            if len(neg_idx) > 0:
                p = neg_idx[torch.randint(len(neg_idx), (n_neg,), device=neg_idx.device)]
                l = torch.zeros(p.shape[0], device=device)
            else:
                p = torch.zeros((1, 2), dtype=torch.long, device=device)
                l = torch.zeros(1, device=device)
        xy = torch.stack([p[:, 1], p[:, 0]], dim=-1).float()
        pts.append(xy[None, ...])
        labels.append(l[None, ...])
    max_pts = max(x.size(1) for x in pts)
    bpts, blbl = [], []
    for xy, l in zip(pts, labels):
        if xy.size(1) < max_pts:
            pad = max_pts - xy.size(1)
            xy = F.pad(xy, (0, 0, 0, pad), value=-1.0)
            l = F.pad(l, (0, pad), value=-1.0)
        bpts.append(xy)
        blbl.append(l)
    return torch.stack(bpts, 0), torch.stack(blbl, 0)


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    out = F.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=align_corners)
    if add_dim:
        out = out.squeeze(3)
    return out


def _calc_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    # uncertainty = -|logit|
    return -torch.abs(logits)


def _get_uncertain_point_coords(coarse_logits: torch.Tensor, num_points: int, oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75) -> torch.Tensor:
    assert oversample_ratio >= 1.0 and 0.0 <= importance_sample_ratio <= 1.0
    N, C, H, W = coarse_logits.shape
    num_sampled = int(num_points * oversample_ratio)
    # random coords in [0,1]
    point_coords = torch.rand(N, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)  # [N,C,num_sampled]
    point_uncertainties = _calc_uncertainty(point_logits)  # [N,C,num_sampled]
    # for binary mask C=1
    topk = max(1, int(num_points * importance_sample_ratio))
    idx = torch.topk(point_uncertainties[:, 0, :], k=topk, dim=1)[1]
    shift = num_sampled * torch.arange(N, dtype=torch.long, device=coarse_logits.device)[:, None]
    idx = (idx + shift).view(-1)
    coords_topk = point_coords.view(-1, 2)[idx].view(N, topk, 2)
    num_random = num_points - topk
    if num_random > 0:
        rand_coords = torch.rand(N, num_random, 2, device=coarse_logits.device)
        coords = torch.cat([coords_topk, rand_coords], dim=1)
    else:
        coords = coords_topk
    return coords  # [N,P,2]


def compute_metrics(logits_b1hw, target_b1hw, thr=0.5):
    prob = torch.sigmoid(logits_b1hw)
    pred = (prob >= thr).float()
    target = target_b1hw.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union, torch.ones_like(union))
    tp = inter
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    precision = torch.where((tp + fp) > 0, tp / (tp + fp), torch.ones_like(tp))
    recall = torch.where((tp + fn) > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    return iou.mean().item(), f1.mean().item()


def format_metric_tag(epoch: int, miou: float, niou: float, f1: float, pd: float, fa: float, thr: float = None):
    def fmt(name: str, value: float, scale: float = 1.0):
        scaled = value * scale
        return f"{name}{scaled:.2f}".replace(".", "p")
    parts = [
        f"ep{epoch:03d}",
        fmt("miou", miou, 100.0),
        fmt("niou", niou, 100.0),
        fmt("f1", f1, 100.0),
        fmt("pd", pd, 100.0),
        fmt("fa", fa, 1e6),
    ]
    if thr is not None:
        parts.append(fmt("thr", thr, 100.0))
    return "_".join(parts)


def log_line(message: str, log_path: str = None):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    print(message)


def _dedup_trainable_params(params):
    out = []
    seen = set()
    for p in params:
        if p is None or (not getattr(p, "requires_grad", False)):
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _exclude_params(params, exclude_params):
    exclude_ids = {id(p) for p in exclude_params}
    out = []
    seen = set()
    for p in params:
        if p is None or (not getattr(p, "requires_grad", False)):
            continue
        pid = id(p)
        if pid in exclude_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _select_topk_points(point_coords: torch.Tensor, point_labels: torch.Tensor, top_k: int):
    bsz, _, _ = point_coords.shape
    top_k = max(1, int(top_k))
    coords_out = torch.full((bsz, top_k, 2), -1.0, device=point_coords.device, dtype=point_coords.dtype)
    labels_out = torch.full((bsz, top_k), -1, device=point_labels.device, dtype=point_labels.dtype)
    for b in range(bsz):
        valid = point_labels[b] >= 0
        coords = point_coords[b][valid]
        if coords.numel() == 0:
            continue
        k = min(top_k, coords.shape[0])
        coords_out[b, :k] = coords[:k]
        labels_out[b, :k] = 1
    return coords_out, labels_out


def _run_pgap_with_text_prior(pgap, images, args, text_prior=None):
    if text_prior is not None and getattr(args, "pgap_text_fuse_internal", False):
        return pgap(
            images,
            text_prior=text_prior,
            text_fuse_weight=float(getattr(args, "pgap_text_fuse_weight", 0.5)),
            text_fuse_mode=str(getattr(args, "pgap_text_fuse_mode", "mul")),
        )
    return pgap(images)


def _build_pgap_prompts(pgap, images, masks, args, text_prior=None):
    pgap_pts, pgap_lbl, saliency = _run_pgap_with_text_prior(
        pgap, images, args, text_prior=text_prior
    )
    if getattr(args, "pgap_label_by_gt", False):
        pgap_pts, pgap_lbl = pgap.label_points_by_gt(
            pgap_pts,
            pgap_lbl,
            masks,
            saliency_map=saliency,
            min_pos=args.pgap_min_pos,
            max_neg=args.pgap_max_neg,
        )
    return pgap_pts, pgap_lbl, saliency


def _merge_dense_mask_prompts(base_prompt, text_prompt, alpha: float):
    if text_prompt is None:
        return base_prompt
    if base_prompt is None:
        return text_prompt
    a = max(0.0, min(1.0, float(alpha)))
    return (1.0 - a) * base_prompt + a * text_prompt


def _select_text_sparse_prompt_source(
    args,
    raw_clip_feat,
    fused_clip_feat,
    fused_clip_token_feat=None,
    fused_clip_token_mask=None,
):
    source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
    if source == "raw_global":
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        return None, None
    if source == "fused_global":
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        return None, None
    if source == "fused_tokens":
        if fused_clip_token_feat is not None:
            return fused_clip_token_feat, fused_clip_token_mask
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        return None, None
    raise ValueError(f"Unsupported text_sparse_prompt_source: {source}")


def _build_text_prompt_inputs(
    model,
    args,
    img_emb,
    clip_feat,
    raw_clip_feat=None,
    clip_token_feat=None,
    clip_token_mask=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
):
    if raw_clip_feat is None and clip_feat is None and clip_token_feat is None:
        return None, None
    sparse_prompt = None
    dense_prompt = None
    if text_sparse_prompt is not None:
        sparse_source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
        sparse_input, sparse_mask = _select_text_sparse_prompt_source(
            args,
            raw_clip_feat=raw_clip_feat,
            fused_clip_feat=clip_feat,
            fused_clip_token_feat=clip_token_feat,
            fused_clip_token_mask=clip_token_mask,
        )
        if sparse_input is not None:
            if sparse_input.dim() == 3:
                sparse_prompt = text_sparse_prompt(sparse_input, attention_mask=sparse_mask)
            else:
                sparse_prompt = text_sparse_prompt(
                    sparse_input,
                    use_global_prompt_enhance=(sparse_source == "raw_global"),
                )
    if text_dense_prompt is not None:
        target_size = getattr(model.prompt_encoder, "mask_input_size", None)
        if getattr(text_dense_prompt, "expects_token_level", False):
            dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
            if dense_text_input is not None:
                dense_prompt = text_dense_prompt(
                    img_emb,
                    dense_text_input,
                    attention_mask=clip_token_mask if clip_token_feat is not None else None,
                    output_size=tuple(target_size) if target_size is not None else None,
                )
        elif clip_feat is not None:
            dense_prompt = text_dense_prompt(
                img_emb,
                clip_feat,
                output_size=tuple(target_size) if target_size is not None else None,
            )
        if dense_prompt is not None:
            dense_prompt = dense_prompt * float(getattr(args, "text_dense_prompt_scale", 1.0))
    return sparse_prompt, dense_prompt


def _build_bifusion_text_inputs(
    clip_feat: Optional[torch.Tensor],
    clip_token_feat: Optional[torch.Tensor],
    clip_token_mask: Optional[torch.Tensor],
):
    if clip_token_feat is not None:
        if clip_token_mask is None:
            clip_token_mask = torch.ones(
                (clip_token_feat.shape[0], clip_token_feat.shape[1]),
                device=clip_token_feat.device,
                dtype=torch.long,
            )
        return clip_token_feat, clip_token_mask
    if clip_feat is not None:
        return clip_feat.unsqueeze(1), torch.ones(
            (clip_feat.shape[0], 1),
            device=clip_feat.device,
            dtype=torch.long,
        )
    return None, None


def _masked_text_mean(text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if text_mask is None:
        return text_tokens.mean(dim=1)
    mask = (text_mask > 0).to(text_tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (text_tokens * mask).sum(dim=1) / denom


def _apply_backbone_bifusion_adapter(
    model,
    backbone_bifusion_adapter,
    images,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
):
    if backbone_bifusion_adapter is None:
        img_emb, interms = model.get_image_embeddings(images)
        return img_emb, interms, clip_feat, clip_token_feat, clip_token_mask
    text_tokens, text_mask = _build_bifusion_text_inputs(
        clip_feat=clip_feat,
        clip_token_feat=clip_token_feat,
        clip_token_mask=clip_token_mask,
    )
    if text_tokens is None or not hasattr(model, "get_image_embeddings_with_text"):
        img_emb, interms = model.get_image_embeddings(images)
        return img_emb, interms, clip_feat, clip_token_feat, clip_token_mask
    img_emb, interms, text_tokens_out, text_mask_out = model.get_image_embeddings_with_text(
        images,
        text_tokens,
        text_attention_mask=text_mask,
    )
    text_global_out = _masked_text_mean(text_tokens_out, text_mask_out)
    return img_emb, interms, text_global_out, text_tokens_out, text_mask_out


def _apply_bifusion_adapter(
    bifusion_adapter,
    img_emb,
    interms,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
):
    if bifusion_adapter is None:
        return img_emb, interms, clip_feat, clip_token_feat, clip_token_mask
    text_tokens, text_mask = _build_bifusion_text_inputs(
        clip_feat=clip_feat,
        clip_token_feat=clip_token_feat,
        clip_token_mask=clip_token_mask,
    )
    if text_tokens is None:
        return img_emb, interms, clip_feat, clip_token_feat, clip_token_mask

    img_emb, interms, text_tokens_out, text_mask_out, text_global_out = bifusion_adapter(
        img_emb,
        interms,
        text_tokens,
        attention_mask=text_mask,
    )
    return img_emb, interms, text_global_out, text_tokens_out, text_mask_out


def _build_pgap_text_prior(
    model,
    args,
    img_emb,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
    text_dense_prompt=None,
    output_size=None,
):
    if not getattr(args, "pgap_text_fuse_internal", False):
        return None
    if text_dense_prompt is None:
        return None
    if getattr(text_dense_prompt, "expects_token_level", False):
        dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
        if dense_text_input is None:
            return None
        prior = text_dense_prompt(
            img_emb,
            dense_text_input,
            attention_mask=clip_token_mask if clip_token_feat is not None else None,
            output_size=tuple(output_size) if output_size is not None else None,
        )
        if prior is not None:
            prior = prior * float(getattr(args, "text_dense_prompt_scale", 1.0))
        return prior
    if clip_feat is None:
        return None
    prior = text_dense_prompt(
        img_emb,
        clip_feat,
        output_size=tuple(output_size) if output_size is not None else None,
    )
    if prior is not None:
        prior = prior * float(getattr(args, "text_dense_prompt_scale", 1.0))
    return prior


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    pgap=None,
    fab_criterion=None,
    scr_criterion=None,
    text_conditioner=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
    bifusion_adapter=None,
    backbone_bifusion_adapter=None,
):
    model.train()
    if pgap is not None:
        pgap.train()
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    bce = nn.BCEWithLogitsLoss()
    nwd_weight = float(getattr(args, "nwd_weight", 0.0))
    nwd_criterion = NWDLoss(constant=float(getattr(args, "nwd_constant", 12.0))).to(device) if nwd_weight > 0.0 else None
    meter_loss, n = 0.0, 0
    skipped_nonfinite = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape

        with autocast_ctx(device):
            clip_feat = None
            raw_clip_feat = None
            clip_token_feat = None
            clip_token_mask = None
            if "clip_text_feat" in batch and (
                text_conditioner is not None
                or text_sparse_prompt is not None
                or text_dense_prompt is not None
                or bifusion_adapter is not None
                or backbone_bifusion_adapter is not None
            ):
                clip_feat = batch["clip_text_feat"].to(device, non_blocking=True)
                raw_clip_feat = clip_feat
            if (text_sparse_prompt is not None or bifusion_adapter is not None or backbone_bifusion_adapter is not None) and "clip_text_token_feat" in batch:
                clip_token_feat = batch["clip_text_token_feat"].to(device, non_blocking=True)
                if "clip_text_attn_mask" in batch:
                    clip_token_mask = batch["clip_text_attn_mask"].to(device, non_blocking=True)
            if (not pgap_text_prior_only) and backbone_bifusion_adapter is not None:
                img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                    model=model,
                    backbone_bifusion_adapter=backbone_bifusion_adapter,
                    images=images,
                    clip_feat=clip_feat,
                    clip_token_feat=clip_token_feat,
                    clip_token_mask=clip_token_mask,
                )
            else:
                img_emb, interms = model.get_image_embeddings(images)
        if (not pgap_text_prior_only) and bifusion_adapter is not None:
            img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_bifusion_adapter(
                bifusion_adapter=bifusion_adapter,
                img_emb=img_emb,
                interms=interms,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        if (not pgap_text_prior_only) and text_conditioner is not None and clip_feat is not None:
            img_emb = text_conditioner(img_emb, clip_feat)
        pgap_text_prior = None
        if pgap is not None:
            pgap_text_prior = _build_pgap_text_prior(
                model,
                args,
                img_emb,
                clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_dense_prompt=text_dense_prompt,
                output_size=(H, W),
            )
        mask_prompt = None
        if pgap is not None:
            pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(
                pgap, images, masks, args, text_prior=pgap_text_prior
            )
            if args.use_feature_mod:
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pts = pgap_pts.unsqueeze(1)
                lbl = pgap_lbl.unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
            else:
                pts, lbl = sample_points_from_mask(
                    masks,
                    n_pos=args.n_pos,
                    n_neg=args.n_neg,
                    boundary_prior=bool(args.boundary_prior_sampling),
                    boundary_ratio=float(args.boundary_ratio),
                )
                pts, lbl = pts.to(device), lbl.to(device)
        else:
            pts, lbl = sample_points_from_mask(
                masks,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                boundary_prior=bool(args.boundary_prior_sampling),
                boundary_ratio=float(args.boundary_ratio),
            )
            pts, lbl = pts.to(device), lbl.to(device)
        if pgap_text_prior_only:
            text_sparse_embeds, text_dense_mask = None, None
            mask_prompt_eff = mask_prompt
        else:
            text_sparse_embeds, text_dense_mask = _build_text_prompt_inputs(
                model, args, img_emb, clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                text_dense_mask,
                getattr(args, "text_dense_prompt_merge_alpha", 0.5),
            )
        # HQ warmup: force using only HQ mask during early epochs
        use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
        pred_masks, _ = model.predict_masks(
            img_emb,
            interms,
            pts,
            lbl,
            batched_masks=mask_prompt_eff,
            text_sparse_embeddings=text_sparse_embeds,
            multimask_output=False,
            input_h=H,
            input_w=W,
            output_h=H,
            output_w=W,
            hq_token_only=use_hq_only,
        )
        logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
        if not torch.isfinite(logits).all():
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            log_line(
                f"[warn] Skip non-finite logits at epoch {epoch:03d}, batch {batch_idx}",
                args.log_file,
            )
            continue
        loss = bce(logits, masks.unsqueeze(1)) + dice_loss(logits, masks.unsqueeze(1))
        if nwd_criterion is not None:
            loss = loss + nwd_weight * nwd_criterion(logits, masks.unsqueeze(1))
        freq_weight = float(getattr(args, "freq_consistency_weight", 0.0))
        if freq_weight > 0.0:
            bins = max(2, int(getattr(args, "freq_consistency_bins", 32)))
            pred_prob = torch.sigmoid(logits)
            gt_mask = masks.unsqueeze(1).float()
            pred_profile = radial_frequency_profile(pred_prob, bins)
            gt_profile = radial_frequency_profile(gt_mask, bins)
            freq_loss = torch.mean(torch.abs(pred_profile - gt_profile))
            loss = loss + freq_weight * freq_loss
        # FAB Loss (Frequency-Aware Boundary Loss)
        if fab_criterion is not None:
            fab_loss_val = fab_criterion(logits, masks.unsqueeze(1))
            loss = loss + args.fab_weight * fab_loss_val
        # SCR Loss (Signal-to-Clutter Ratio Loss)
        if scr_criterion is not None:
            scr_loss_val = scr_criterion(logits, masks.unsqueeze(1), images)
            loss = loss + args.scr_weight * scr_loss_val
        # Prompt-Robust Consistency Loss
        if getattr(args, 'use_prompt_robust_loss', False) and pgap is None:
            # 生成扰动 prompt 点
            with torch.no_grad():
                perturb_std = float(getattr(args, 'prompt_robust_perturb_std', 3.0))
                pts_perturbed = pts.clone()
                noise = torch.randn_like(pts_perturbed.float()) * perturb_std
                pts_perturbed = pts_perturbed + noise
                # Clamp to valid image range
                pts_perturbed[..., 0] = pts_perturbed[..., 0].clamp(0, W - 1)
                pts_perturbed[..., 1] = pts_perturbed[..., 1].clamp(0, H - 1)
            text_sparse_embeds_p, text_dense_mask_p = _build_text_prompt_inputs(
                model, args, img_emb.detach(), clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            pred_masks_p, _ = model.predict_masks(
                img_emb.detach(),  # detach to avoid double backward through encoder
                interms,
                pts_perturbed,
                lbl,
                batched_masks=_merge_dense_mask_prompts(
                    mask_prompt,
                    text_dense_mask_p,
                    getattr(args, "text_dense_prompt_merge_alpha", 0.5),
                ),
                text_sparse_embeddings=text_sparse_embeds_p,
                multimask_output=False,
                input_h=H, input_w=W,
                output_h=H, output_w=W,
                hq_token_only=use_hq_only,
            )
            logits_p = pred_masks_p[:, 0, 0, ...].unsqueeze(1)
            # 一致性损失: 两次预测应该相似 (Dice)
            prob_clean = torch.sigmoid(logits.detach())
            prob_perturb = torch.sigmoid(logits_p)
            inter_c = (prob_clean * prob_perturb).sum(dim=(1, 2, 3))
            denom_c = prob_clean.sum(dim=(1, 2, 3)) + prob_perturb.sum(dim=(1, 2, 3))
            consist_dice = 1 - ((2 * inter_c + 1.0) / (denom_c + 1.0)).mean()
            # 扰动预测也应对齐 GT
            consist_bce = F.binary_cross_entropy_with_logits(logits_p, masks.unsqueeze(1))
            prompt_robust_w = float(getattr(args, 'prompt_robust_weight', 0.1))
            loss = loss + prompt_robust_w * (consist_dice + 0.5 * consist_bce)

        if args.use_point_loss:
            # point-based loss on uncertain points
            num_points = args.point_loss_points
            coords = _get_uncertain_point_coords(logits.detach(), num_points=num_points,
                                                 oversample_ratio=args.point_loss_oversample,
                                                 importance_sample_ratio=args.point_loss_importance)
            gt_points = point_sample(masks.unsqueeze(1).float(), coords, align_corners=False)  # [B,1,P]
            pr_points = point_sample(logits, coords, align_corners=False)  # [B,1,P]
            # BCE at points
            bce_points = F.binary_cross_entropy_with_logits(pr_points, gt_points, reduction='none').mean(1).mean()
            # Dice at points (use same formula as image-wise)
            pr_sig = torch.sigmoid(pr_points)
            num = 2 * (pr_sig * gt_points).sum(dim=2)
            den = pr_sig.sum(dim=2) + gt_points.sum(dim=2)
            dice_pts = 1 - (num + 1) / (den + 1)
            dice_pts = dice_pts.mean()
            loss = loss + args.point_loss_weight * (bce_points + dice_pts)

        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            log_line(
                f"[warn] Skip non-finite loss at epoch {epoch:03d}, batch {batch_idx}",
                args.log_file,
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter_loss += loss.item() * B
        n += B
    if skipped_nonfinite > 0:
        log_line(
            f"[warn] Skipped {skipped_nonfinite} non-finite train batches at epoch {epoch:03d}",
            args.log_file,
        )
    return meter_loss / max(n, 1)


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    args,
    epoch: int,
    pgap=None,
    text_conditioner=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
    bifusion_adapter=None,
    backbone_bifusion_adapter=None,
):
    model.eval()
    if pgap is not None:
        pgap.eval()
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    # 使用备份版的批次平均方式计算IoU和F1
    ious, f1s = [], []
    niou_sum = 0.0
    niou_count = 0
    thr_sum = 0.0
    thr_count = 0
    pd_fa = PD_FA(distance_thresh=getattr(args, "pd_fa_dist", 3))
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape
        clip_feat = None
        raw_clip_feat = None
        clip_token_feat = None
        clip_token_mask = None
        if "clip_text_feat" in batch and (
            text_conditioner is not None
            or text_sparse_prompt is not None
            or text_dense_prompt is not None
            or bifusion_adapter is not None
            or backbone_bifusion_adapter is not None
        ):
            clip_feat = batch["clip_text_feat"].to(device, non_blocking=True)
            raw_clip_feat = clip_feat
        if (text_sparse_prompt is not None or bifusion_adapter is not None or backbone_bifusion_adapter is not None) and "clip_text_token_feat" in batch:
            clip_token_feat = batch["clip_text_token_feat"].to(device, non_blocking=True)
            if "clip_text_attn_mask" in batch:
                clip_token_mask = batch["clip_text_attn_mask"].to(device, non_blocking=True)
        if (not pgap_text_prior_only) and backbone_bifusion_adapter is not None:
            img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                model=model,
                backbone_bifusion_adapter=backbone_bifusion_adapter,
                images=images,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        else:
            img_emb, interms = model.get_image_embeddings(images)
        if (not pgap_text_prior_only) and bifusion_adapter is not None:
            img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_bifusion_adapter(
                bifusion_adapter=bifusion_adapter,
                img_emb=img_emb,
                interms=interms,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        if (not pgap_text_prior_only) and text_conditioner is not None and clip_feat is not None:
            img_emb = text_conditioner(img_emb, clip_feat)
        pgap_text_prior = None
        if pgap is not None:
            pgap_text_prior = _build_pgap_text_prior(
                model,
                args,
                img_emb,
                clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_dense_prompt=text_dense_prompt,
                output_size=(H, W),
            )
        mask_prompt = None
        if pgap is not None:
            if getattr(args, "pgap_two_stage", False):
                pgap_pts, pgap_lbl, saliency = _run_pgap_with_text_prior(
                    pgap, images, args, text_prior=pgap_text_prior
                )
                if args.use_feature_mod:
                    img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pos_pts, pos_lbl = _select_topk_points(pgap_pts, pgap_lbl, args.pgap_stage1_top_k)
                pts1 = pos_pts.unsqueeze(1).to(device)
                lbl1 = pos_lbl.unsqueeze(1).to(device)
                use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
                if pgap_text_prior_only:
                    text_sparse_stage1, text_dense_stage1 = None, None
                else:
                    text_sparse_stage1, text_dense_stage1 = _build_text_prompt_inputs(
                        model, args, img_emb, clip_feat,
                        raw_clip_feat=raw_clip_feat,
                        clip_token_feat=clip_token_feat,
                        clip_token_mask=clip_token_mask,
                        text_sparse_prompt=text_sparse_prompt,
                        text_dense_prompt=text_dense_prompt,
                    )
                pred_masks1, _ = model.predict_masks(
                    img_emb,
                    interms,
                    pts1,
                    lbl1,
                    batched_masks=_merge_dense_mask_prompts(
                        mask_prompt,
                        text_dense_stage1,
                        getattr(args, "text_dense_prompt_merge_alpha", 0.5),
                    ),
                    text_sparse_embeddings=text_sparse_stage1,
                    multimask_output=False,
                    input_h=H,
                    input_w=W,
                    output_h=H,
                    output_w=W,
                    hq_token_only=use_hq_only,
                )
                logits1 = pred_masks1[:, 0, 0, ...].unsqueeze(1)
                coarse = (torch.sigmoid(logits1) >= args.pgap_stage1_thr).float()
                neg_pts, neg_lbl = pgap.select_negatives_from_mask(
                    pgap_pts, pgap_lbl, saliency, coarse[:, 0], args.pgap_stage2_neg
                )
                pts = torch.cat([pos_pts, neg_pts], dim=1).unsqueeze(1)
                lbl = torch.cat([pos_lbl, neg_lbl], dim=1).unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
            else:
                pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(
                    pgap, images, masks, args, text_prior=pgap_text_prior
                )
                if args.use_feature_mod:
                    img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pts = pgap_pts.unsqueeze(1)
                lbl = pgap_lbl.unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
        else:
            pts, lbl = sample_points_from_mask(
                masks,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                boundary_prior=bool(args.boundary_prior_sampling),
                boundary_ratio=float(args.boundary_ratio),
            )
            pts, lbl = pts.to(device), lbl.to(device)
        if pgap_text_prior_only:
            text_sparse_embeds, text_dense_mask = None, None
            mask_prompt_eff = mask_prompt
        else:
            text_sparse_embeds, text_dense_mask = _build_text_prompt_inputs(
                model, args, img_emb, clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                text_dense_mask,
                getattr(args, "text_dense_prompt_merge_alpha", 0.5),
            )
        use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
        pred_masks, _ = model.predict_masks(
            img_emb,
            interms,
            pts,
            lbl,
            batched_masks=mask_prompt_eff,
            text_sparse_embeddings=text_sparse_embeds,
            multimask_output=False,
            input_h=H,
            input_w=W,
            output_h=H,
            output_w=W,
            hq_token_only=use_hq_only,
        )
        logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
        if args.val_thr_search:
            best_iou, best_thr = -1.0, args.thr
            thr = args.val_thr_min
            while thr <= args.val_thr_max + 1e-6:
                miou_t, _ = compute_metrics(logits, masks.unsqueeze(1), thr=thr)
                if miou_t > best_iou:
                    best_iou, best_thr = miou_t, thr
                thr += args.val_thr_step
            thr_used = best_thr
        else:
            thr_used = args.thr

        # 使用备份版的计算方式：每批次调用compute_metrics
        miou, mf1 = compute_metrics(logits, masks.unsqueeze(1), thr=thr_used)
        ious.append(miou)
        f1s.append(mf1)

        # 保留nIoU计算（样本级平均）
        prob = torch.sigmoid(logits)
        pred = (prob >= thr_used).float()
        target = masks.unsqueeze(1).float()
        inter_s = (pred * target).sum(dim=(1, 2, 3))
        union_s = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou_s = torch.where(union_s > 0, inter_s / union_s, torch.ones_like(union_s))
        niou_sum += iou_s.sum().item()
        niou_count += int(iou_s.numel())

        # 保留PD/FA计算
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        for b in range(pred_cpu.shape[0]):
            pd_fa.update(pred_cpu[b, 0], target_cpu[b, 0], (H, W))

        thr_sum += float(thr_used)
        thr_count += 1

    # 备份版方式：批次平均
    miou_avg = sum(ious) / len(ious) if ious else 0.0
    f1_avg = sum(f1s) / len(f1s) if f1s else 0.0
    niou_avg = niou_sum / niou_count if niou_count > 0 else 0.0
    pd_val, fa_val = pd_fa.get()
    thr_used = (thr_sum / thr_count) if thr_count > 0 else args.thr
    return miou_avg, niou_avg, f1_avg, pd_val, fa_val, thr_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_txt", type=str, default="train.txt")
    p.add_argument("--val_txt", type=str, default="test.txt")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--keep_ratio_pad", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n_pos", type=int, default=4)
    p.add_argument("--n_neg", type=int, default=4)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--out_dir", type=str, default="./outputs_sam_sirst_hq")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--model", type=str, default="vitt", choices=["vitt", "vits"])  # kept for naming
    p.add_argument("--hq_token_only", action="store_true")
    p.add_argument("--hq_warmup_epochs", type=int, default=0,
                   help="If >0, use HQ token only for the first N epochs.")
    p.add_argument("--init_from_baseline", type=str, default=None,
                   help="Optional path to EfficientSAM baseline checkpoint to partially initialize from.")
    p.add_argument("--use_fs_adapter", action="store_true",
                   help="Enable frequency-spatial adapter inside ViT blocks.")
    p.add_argument("--use_ms_fusion", action="store_true",
                   help="Enable multi-scale fusion from intermediate ViT blocks.")
    p.add_argument("--use_detail_enhancer", action="store_true",
                   help="Enable Sobel detail enhancer on shallow features.")
    p.add_argument("--early_exit_layer", type=int, default=0,
                   help="Exit after N transformer blocks (1-based). Use 0 to disable.")
    # Radial gate for HQ-SAM (optional)
    p.add_argument("--use_radial_gate_hq", action="store_true", help="Enable RadialFreqGate for HQ-SAM.")
    p.add_argument("--rgate_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply radial gate: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--freq_patch_size_hq", type=int, default=8)
    p.add_argument("--radial_bins_hq", type=int, default=6)
    p.add_argument("--radial_channel_shared_hq", action="store_true")
    p.add_argument("--rgate_strength_enc", type=float, default=0.5)
    p.add_argument("--rgate_strength_dec", type=float, default=0.5)
    p.add_argument("--rgate_edge_boost", type=float, default=0.5,
                   help="Edge-aware high-frequency boost factor for RadialFreqGate (set 0 to disable).")
    p.add_argument("--rgate_high_freq_thresh", type=float, default=0.6,
                   help="Normalized radial threshold above which frequencies are treated as high.")
    # ASG gate for HQ-SAM (optional)
    p.add_argument("--use_asg_hq", action="store_true", help="Enable AnisotropicSpectralGating for HQ-SAM.")
    p.add_argument("--asg_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply ASG: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--asg_radial_bins", type=int, default=64)
    p.add_argument("--asg_angular_bins", type=int, default=128)
    p.add_argument("--asg_variant", type=str, default="asg1", choices=["asg1", "asg2"],
                   help="Which ASG implementation to use.")
    p.add_argument("--asg_strength_enc", type=float, default=1.0)
    p.add_argument("--asg_strength_dec", type=float, default=1.0)
    # AFD (Adaptive Frequency Decomposition) module for HQ-SAM
    p.add_argument("--use_afd_hq", action="store_true", help="Enable AdaptiveFrequencyDecomposition for HQ-SAM.")
    p.add_argument("--afd_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply AFD: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--afd_patch_size", type=int, default=8,
                   help="Patch size for AFD FFT processing.")
    p.add_argument("--afd_num_bins", type=int, default=16,
                   help="Number of discrete bins for cutoff prediction.")
    p.add_argument("--afd_low_ratio", type=float, default=1.0,
                   help="Low frequency gain initial value (learnable, default 1.0 = no suppression).")
    p.add_argument("--afd_high_ratio", type=float, default=1.0,
                   help="High frequency gain initial value (learnable, default 1.0 = no enhancement).")
    p.add_argument("--afd_learnable_gains", action="store_true", default=True,
                   help="Use learnable gains instead of fixed ratios (default True).")
    p.add_argument("--afd_fixed_gains", action="store_true",
                   help="Use fixed gains (disable learnable gains).")
    p.add_argument("--afd_channel_wise", action="store_true",
                   help="Use per-channel independent gains instead of global gains.")
    p.add_argument("--afd_strength_enc", type=float, default=0.5,
                   help="Residual strength for AFD at encoder.")
    p.add_argument("--afd_strength_dec", type=float, default=0.5,
                   help="Residual strength for AFD at decoder.")
    # MSFE (Multi-Scale Frequency Enhancement) module
    p.add_argument("--use_msfe_hq", action="store_true",
                   help="Enable MultiScaleFrequencyEnhancement for HQ-SAM.")
    p.add_argument("--msfe_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply MSFE: encoder, decoder, or both.")
    p.add_argument("--msfe_patch_sizes", type=str, default="4,8,16",
                   help="Comma-separated patch sizes, e.g., '4,8,16'.")
    p.add_argument("--msfe_num_bins", type=int, default=8,
                   help="Number of radial frequency bins per scale.")
    p.add_argument("--msfe_fusion", type=str, default="attention", choices=["attention", "concat", "average"],
                   help="Fusion method for multi-scale outputs.")
    p.add_argument("--msfe_strength_enc", type=float, default=0.5,
                   help="Residual strength for MSFE at encoder.")
    p.add_argument("--msfe_strength_dec", type=float, default=0.5,
                   help="Residual strength for MSFE at decoder.")
    p.add_argument("--freq_consistency_weight", type=float, default=0.0,
                   help="Weight for radial frequency consistency loss.")
    p.add_argument("--freq_consistency_bins", type=int, default=32,
                   help="Radial bins used for frequency consistency loss.")
    # FAB Loss (Frequency-Aware Boundary Loss)
    p.add_argument("--use_fab_loss", action="store_true",
                   help="Enable Frequency-Aware Boundary Loss for small target detection.")
    p.add_argument("--fab_weight", type=float, default=0.5,
                   help="Weight for FAB Loss.")
    p.add_argument("--fab_num_bins", type=int, default=16,
                   help="Number of radial frequency bins for FAB Loss.")
    p.add_argument("--fab_boundary_width", type=int, default=3,
                   help="Boundary extraction kernel size.")
    p.add_argument("--fab_high_freq_weight", type=float, default=2.0,
                   help="Weight multiplier for high frequency components.")
    # SCR Loss (Signal-to-Clutter Ratio Loss)
    p.add_argument("--use_scr_loss", action="store_true",
                   help="Enable Signal-to-Clutter Ratio Loss for IRSTD.")
    p.add_argument("--scr_weight", type=float, default=0.1,
                   help="Weight for SCR Loss.")
    p.add_argument("--scr_inner_k", type=int, default=5,
                   help="Inner annular dilation kernel size for SCR.")
    p.add_argument("--scr_outer_k", type=int, default=15,
                   help="Outer annular dilation kernel size for SCR.")
    # Prompt-Robust Consistency Loss
    p.add_argument("--use_prompt_robust_loss", action="store_true",
                   help="Enable Prompt-Robustness Consistency Loss.")
    p.add_argument("--prompt_robust_weight", type=float, default=0.1,
                   help="Weight for prompt robustness consistency loss.")
    p.add_argument("--prompt_robust_perturb_std", type=float, default=3.0,
                   help="Std of Gaussian noise added to prompt points (pixels).")
    # Proposed options (default OFF)
    p.add_argument("--use_point_loss", action="store_true",
                   help="Enable uncertainty-based point sampling BCE+Dice as auxiliary loss.")
    p.add_argument("--point_loss_points", type=int, default=4096,
                   help="Number of points for point loss.")
    p.add_argument("--point_loss_oversample", type=float, default=3.0,
                   help="Oversample ratio for uncertain point selection.")
    p.add_argument("--point_loss_importance", type=float, default=0.75,
                   help="Importance sample ratio for uncertain points.")
    p.add_argument("--point_loss_weight", type=float, default=0.3,
                   help="Weight for point loss term.")
    p.add_argument("--nwd_weight", type=float, default=0.0,
                   help="Weight for NWD loss (0 to disable).")
    p.add_argument("--nwd_constant", type=float, default=12.0,
                   help="Normalization constant for NWD loss.")
    p.add_argument("--boundary_prior_sampling", action="store_true",
                   help="Prefer sampling points near GT boundary.")
    p.add_argument("--boundary_ratio", type=float, default=0.5,
                   help="Fraction of pos/neg points sampled from boundary region.")
    # Task Tokens (Learnable Prompt Tokens for IRSTD)
    p.add_argument("--use_task_tokens", action="store_true",
                   help="Enable learnable task tokens in PromptEncoder for IRSTD prior.")
    p.add_argument("--num_task_tokens", type=int, default=2,
                   help="Number of learnable task tokens (1-4 recommended).")
    p.add_argument("--task_token_init_scale", type=float, default=0.02,
                   help="Initialization scale for task tokens (small to avoid disrupting pretrain).")
    # Phase prompt generator (PGAP)
    p.add_argument("--use_pgap", action="store_true",
                   help="Use PhasePromptGenerator to auto-generate prompt points.")
    p.add_argument("--pgap_top_k", type=int, default=5)
    p.add_argument("--pgap_min_dist", type=int, default=10)
    p.add_argument("--pgap_saliency_thr", type=float, default=0.1)
    p.add_argument("--pgap_blur_kernel", type=int, default=5)
    p.add_argument("--pgap_blur_sigma", type=float, default=1.0)
    p.add_argument("--pgap_border_width", type=int, default=12)
    p.add_argument("--pgap_no_window", action="store_true")
    p.add_argument("--pgap_no_dynamic_thr", action="store_true")
    p.add_argument("--pgap_dyn_quantile", type=float, default=0.9)
    p.add_argument("--pgap_dyn_mode", type=str, default="max", choices=["max", "replace"])
    p.add_argument("--pgap_no_dynamic_topk", action="store_true")
    p.add_argument("--pgap_min_top_k", type=int, default=1)
    p.add_argument("--pgap_use_dct", action="store_true")
    p.add_argument("--pgap_text_fuse_internal", action="store_true",
                   help="Fuse text dense prior into PGAP saliency inside PGAP before point extraction.")
    p.add_argument("--pgap_text_fuse_weight", type=float, default=0.5,
                   help="Fusion strength for internal PGAP-text saliency fusion.")
    p.add_argument("--pgap_text_fuse_mode", type=str, default="mul", choices=["mul", "add"],
                   help="Internal PGAP-text saliency fusion mode.")
    p.add_argument("--pgap_text_prior_only", action="store_true",
                   help="Pure PGAP-text mode: text is only used to build PGAP internal fused saliency, not injected into SAM (no FiLM/sparse/dense prompt injection).")
    p.add_argument("--use_feature_mod", action="store_true",
                   help="Use PGAP saliency to modulate image embeddings.")
    p.add_argument("--pgap_label_by_gt", action="store_true",
                   help="Use GT to relabel PGAP points: inside=pos, outside=neg.")
    p.add_argument("--pgap_min_pos", type=int, default=1)
    p.add_argument("--pgap_max_neg", type=int, default=2)
    p.add_argument("--pgap_two_stage", action="store_true",
                   help="Two-stage prompting in validation: pos first, then add negatives outside coarse mask.")
    p.add_argument("--pgap_stage1_top_k", type=int, default=1)
    p.add_argument("--pgap_stage1_thr", type=float, default=0.5)
    p.add_argument("--pgap_stage2_neg", type=int, default=2)
    p.add_argument("--use_mask_prompt", action="store_true",
                   help="Use PGAP saliency map as dense mask prompt.")
    p.add_argument("--val_thr_search", action="store_true",
                   help="Enable validation threshold grid search.")
    p.add_argument("--val_thr_min", type=float, default=0.35)
    p.add_argument("--val_thr_max", type=float, default=0.55)
    p.add_argument("--val_thr_step", type=float, default=0.05)
    p.add_argument("--pd_fa_dist", type=int, default=3,
                   help="Distance threshold for PD/FA metrics (in pixels).")
    p.add_argument("--log_file", type=str, default=None,
                   help="Path to log file (default: <out_dir>/log.txt).")
    p.add_argument("--mask_suffix", type=str, default="",
                   help="Optional suffix for mask filenames before extension, e.g. '_pixels0'.")
    # SCTransNet-style preprocessing options (与备份版兼容)
    p.add_argument("--sctransnet_preproc", action="store_true",
                   help="Use SCTransNet-style preprocessing: 16-bit grayscale, dataset normalization, random crop, enhanced augmentation.")
    p.add_argument("--sc_use_noise", action="store_true",
                   help="Add Gaussian noise in SCTransNet augmentation.")
    p.add_argument("--sc_use_gamma", action="store_true",
                   help="Apply random gamma correction in SCTransNet augmentation.")
    p.add_argument("--sc_pos_prob", type=float, default=0.5,
                   help="Probability of cropping region containing target in SCTransNet mode.")
    p.add_argument("--sc_dataset_name", type=str, default=None,
                   help="Dataset name for SCTransNet normalization (auto-detected from data_root if not set).")
    # MLLM text prompt (pre-computed CLIP features)
    p.add_argument("--use_mllm_prompt", action="store_true",
                   help="Enable MLLM-based text prompting with pre-computed CLIP features.")
    p.add_argument("--mllm_features_path", type=str, default="mllm_clip_features.pt",
                   help="Path to pre-computed CLIP text features file (.pt).")
    p.add_argument("--mllm_text_dim", type=int, default=512,
                   help="Dimension of CLIP text features (512 for ViT-B/32).")
    p.add_argument("--disable_text_conditioner", action="store_true",
                   help="Disable FiLM-style text conditioning on image embeddings while keeping other text modules.")
    p.add_argument("--use_text_sparse_prompt", action="store_true",
                   help="Project selected text feature source into extra sparse prompt token(s).")
    p.add_argument("--text_sparse_num_tokens", type=int, default=1,
                   help="Number of text sparse prompt tokens.")
    p.add_argument("--text_sparse_init_scale", type=float, default=0.02,
                   help="Init scale for text sparse base tokens.")
    p.add_argument("--text_sparse_prompt_source", type=str, default="fused_tokens",
                   choices=["raw_global", "fused_global", "fused_tokens"],
                   help="Sparse prompt source: raw_global=original clip_text_feat, fused_global=post-fusion pooled text feature, fused_tokens=post-fusion token features (fallback to fused_global).")
    p.add_argument("--text_sparse_raw_global_gate", action="store_true",
                   help="Enable a small sigmoid gate on the enhanced raw_global sparse prompt delta path.")
    p.add_argument("--text_sparse_raw_global_gate_init_bias", type=float, default=-2.0,
                   help="Init bias for raw_global sparse prompt gate; more negative means weaker initial injection.")
    p.add_argument("--use_text_dense_prompt", action="store_true",
                   help="Generate a text-guided dense mask prompt from image embeddings + CLIP text feature.")
    p.add_argument("--text_dense_hidden_dim", type=int, default=128,
                   help="Hidden channels for text-guided dense mask prompt generator.")
    p.add_argument("--text_dense_prompt_type", type=str, default="global",
                   choices=["global", "token_xattn"],
                   help="Dense text prompt variant: global (v1) or token_xattn (v2 token-level cross-attn).")
    p.add_argument("--text_dense_num_heads", type=int, default=4,
                   help="Number of heads for token_xattn dense prompt variant.")
    p.add_argument("--text_dense_prompt_merge_alpha", type=float, default=0.5,
                   help="Blend ratio when combining PGAP mask prompt and text dense mask prompt.")
    p.add_argument("--text_dense_prompt_scale", type=float, default=1.0,
                   help="Scale factor applied to generated text dense mask prompt before merging.")
    p.add_argument("--use_bifusion_adapter", action="store_true",
                   help="Enable lightweight bidirectional text-vision fusion adapter at two levels (interms + img_emb).")
    p.add_argument("--bifusion_hidden_dim", type=int, default=128,
                   help="Hidden dim for BiFusion attention space.")
    p.add_argument("--bifusion_num_heads", type=int, default=4,
                   help="Number of heads for BiFusion cross-attention.")
    p.add_argument("--bifusion_interms_dim", type=int, default=192,
                   help="Fallback interms channel dim when auto-detection fails.")
    p.add_argument("--bifusion_disable_interms_level", action="store_true",
                   help="Disable interms-level fusion; keep img_emb level only.")
    p.add_argument("--bifusion_img_res_scale", type=float, default=1.0,
                   help="Residual scale for img_emb update in BiFusion.")
    p.add_argument("--bifusion_interms_res_scale", type=float, default=1.0,
                   help="Residual scale for interms update in BiFusion.")
    p.add_argument("--bifusion_text_res_scale", type=float, default=1.0,
                   help="Residual scale for text token update in BiFusion.")
    p.add_argument("--use_bifusion_backbone_blocks", action="store_true",
                   help="Enable bidirectional text-vision fusion inside image encoder blocks.")
    p.add_argument("--use_gated_bifusion_backbone_blocks", action="store_true",
                   help="Enable gated backbone BiFusion for ablation; keeps block-level bidirectional fusion but gates text/vision updates.")
    p.add_argument("--bifusion_block_apply_every", type=int, default=1,
                   help="Apply backbone BiFusion every K encoder blocks.")
    p.add_argument("--bifusion_block_vision_res_scale", type=float, default=1.0,
                   help="Residual scale for vision-token update in backbone BiFusion.")
    p.add_argument("--bifusion_block_text_res_scale", type=float, default=1.0,
                   help="Residual scale for text-token update in backbone BiFusion.")
    p.add_argument("--bifusion_gate_hidden_dim", type=int, default=0,
                   help="Hidden dim for gated backbone BiFusion gates (<=0 uses hidden_dim//4).")
    p.add_argument("--bifusion_gate_init_bias", type=float, default=-2.0,
                   help="Initial bias for gated backbone BiFusion sigmoid gates (negative keeps gates conservative at start).")
    # Freeze/unfreeze strategy configs
    p.add_argument("--freeze_encoder_epochs", type=int, default=-1,
                   help="Freeze image encoder for N epochs first (<=0 to use epochs//4).")
    p.add_argument("--train_prompt_encoder_during_freeze", action="store_true",
                   help="Whether to train prompt encoder during initial freeze stage (default: False).")
    p.add_argument("--freeze_maskdecoder_to_hq", action="store_true", default=True,
                   help="Only train HQ-specific params in MaskDecoder during initial freeze stage.")
    p.add_argument("--unfreeze_all_when_encoder", action="store_true", default=True,
                   help="When unfreezing encoder, also unfreeze full mask decoder and prompt encoder.")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.exp_name is None:
        base = [f"model-{args.model}", f"size-{args.size}", f"bs-{args.batch_size}"]
        auto_name = "_".join(base)
        run_dir = os.path.join(args.out_dir, f"{auto_name}_{ts}")
    else:
        run_dir = os.path.join(args.out_dir, f"{args.exp_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    try:
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    args.out_dir = run_dir
    if args.log_file is None:
        args.log_file = os.path.join(run_dir, "log.txt")
    log_line(f"Run directory: {args.out_dir}", args.log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader = make_loader(
        args.data_root,
        args.train_txt,
        batch_size=args.batch_size,
        size=args.size,
        augment=True,
        keep_ratio_pad=args.keep_ratio_pad,
        workers=args.workers,
        shuffle=True,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=args.sctransnet_preproc,
        sc_use_noise=args.sc_use_noise,
        sc_use_gamma=args.sc_use_gamma,
        sc_pos_prob=args.sc_pos_prob,
        sc_dataset_name=args.sc_dataset_name,
        mllm_features_path=getattr(args, "mllm_features_path", None) if getattr(args, "use_mllm_prompt", False) else None,
    )
    val_loader = make_loader(
        args.data_root,
        args.val_txt,
        batch_size=max(1, args.batch_size // 2),
        size=args.size,
        augment=False,
        keep_ratio_pad=args.keep_ratio_pad,
        workers=args.workers,
        shuffle=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=args.sctransnet_preproc,
        sc_use_noise=False,  # No noise augmentation for validation
        sc_use_gamma=False,  # No gamma augmentation for validation
        sc_pos_prob=args.sc_pos_prob,
        sc_dataset_name=args.sc_dataset_name,
        mllm_features_path=getattr(args, "mllm_features_path", None) if getattr(args, "use_mllm_prompt", False) else None,
    )

    # Model
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=args.init_from_baseline,
        use_adapter=args.use_fs_adapter,
        use_ms_fusion=args.use_ms_fusion,
        use_detail_enhancer=args.use_detail_enhancer,
        early_exit_layer=args.early_exit_layer,
    )
    # Attach frequency gates if requested
    if args.use_asg_hq and args.use_radial_gate_hq:
        log_line("[warn] Both ASG and RadialFreqGate are enabled; using ASG and skipping RadialFreqGate.", args.log_file)
    if args.use_asg_hq:
        try:
            from efficient_sam.asg import AnisotropicSpectralGating, AnisotropicSpectralGating2

            class _ASGDelta(nn.Module):
                # Convert ASG residual output to a delta for x + gate(x) usage.
                def __init__(self, asg):
                    super().__init__()
                    self.asg = asg

                def forward(self, x):
                    return self.asg(x) - x

            if args.asg_variant == "asg2":
                asg_cls = AnisotropicSpectralGating2
                asg_kwargs = {
                    "r_bins": args.asg_radial_bins,
                    "theta_bins": args.asg_angular_bins,
                }
            else:
                asg_cls = AnisotropicSpectralGating
                asg_kwargs = {
                    "num_radial_bins": args.asg_radial_bins,
                    "num_angular_bins": args.asg_angular_bins,
                }

            enc_hw = getattr(model.image_encoder, "image_embedding_size", 64)
            if args.asg_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                asg_enc = asg_cls(dim_enc, enc_hw, enc_hw, **asg_kwargs)
                model.image_encoder.radial_gate = _ASGDelta(asg_enc)
                model.image_encoder.rgate_strength = float(args.asg_strength_enc)
            if args.asg_loc in ("decoder", "both"):
                # hq_features channels = transformer_dim // 8, spatial size = 4x encoder grid
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                hq_hw = enc_hw * 4
                asg_dec = asg_cls(c_dec, hq_hw, hq_hw, **asg_kwargs)
                model.mask_decoder.radial_gate = _ASGDelta(asg_dec)
                model.mask_decoder.rgate_strength_dec = float(args.asg_strength_dec)
        except Exception as e:
            log_line(f"[warn] Failed to attach ASG: {e}", args.log_file)
    elif args.use_radial_gate_hq:
        try:
            from efficient_sam.freq_modules import RadialFreqGate
            if args.rgate_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                model.image_encoder.radial_gate = RadialFreqGate(
                    dim_enc,
                    patch_size=args.freq_patch_size_hq,
                    num_bins=args.radial_bins_hq,
                    channel_shared=args.radial_channel_shared_hq,
                    edge_boost=args.rgate_edge_boost,
                    high_freq_threshold=args.rgate_high_freq_thresh,
                )
                model.image_encoder.rgate_strength = float(args.rgate_strength_enc)
            if args.rgate_loc in ("decoder", "both"):
                # hq_features channels = transformer_dim // 8
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                model.mask_decoder.radial_gate = RadialFreqGate(
                    c_dec,
                    patch_size=args.freq_patch_size_hq,
                    num_bins=args.radial_bins_hq,
                    channel_shared=args.radial_channel_shared_hq,
                    edge_boost=args.rgate_edge_boost,
                    high_freq_threshold=args.rgate_high_freq_thresh,
                )
                model.mask_decoder.rgate_strength_dec = float(args.rgate_strength_dec)
        except Exception as e:
            log_line(f"[warn] Failed to attach RadialFreqGate: {e}", args.log_file)
    # Attach AFD (Adaptive Frequency Decomposition) if requested
    if args.use_afd_hq:
        try:
            from efficient_sam.freq_modules import AdaptiveFrequencyDecomposition
            if args.afd_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                afd_enc = AdaptiveFrequencyDecomposition(
                    dim=dim_enc,
                    patch_size=args.afd_patch_size,
                    num_cutoff_bins=args.afd_num_bins,
                    low_enhance_ratio=args.afd_low_ratio,
                    high_enhance_ratio=args.afd_high_ratio,
                    learnable_gains=not getattr(args, 'afd_fixed_gains', False),
                    channel_wise_gains=getattr(args, 'afd_channel_wise', False),
                )
                model.image_encoder.afd_gate = afd_enc
                model.image_encoder.afd_strength = float(args.afd_strength_enc)
                log_line(f"Attached AFD to encoder (dim={dim_enc}, patch={args.afd_patch_size}, bins={args.afd_num_bins})", args.log_file)
            if args.afd_loc in ("decoder", "both"):
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                afd_dec = AdaptiveFrequencyDecomposition(
                    dim=c_dec,
                    patch_size=args.afd_patch_size,
                    num_cutoff_bins=args.afd_num_bins,
                    low_enhance_ratio=args.afd_low_ratio,
                    high_enhance_ratio=args.afd_high_ratio,
                    learnable_gains=not getattr(args, 'afd_fixed_gains', False),
                    channel_wise_gains=getattr(args, 'afd_channel_wise', False),
                )
                model.mask_decoder.afd_gate = afd_dec
                model.mask_decoder.afd_strength_dec = float(args.afd_strength_dec)
                log_line(f"Attached AFD to decoder (dim={c_dec}, patch={args.afd_patch_size}, bins={args.afd_num_bins})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach AFD: {e}", args.log_file)
    # Attach MSFE (Multi-Scale Frequency Enhancement) if requested
    if getattr(args, "use_msfe_hq", False):
        try:
            from efficient_sam.freq_modules import MultiScaleFrequencyEnhancement
            # Parse patch sizes from comma-separated string
            patch_sizes = tuple(int(x) for x in args.msfe_patch_sizes.split(","))
            if args.msfe_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                msfe_enc = MultiScaleFrequencyEnhancement(
                    dim=dim_enc,
                    patch_sizes=patch_sizes,
                    num_radial_bins=args.msfe_num_bins,
                    fusion_method=args.msfe_fusion,
                )
                model.image_encoder.msfe_gate = msfe_enc
                model.image_encoder.msfe_strength = float(args.msfe_strength_enc)
                log_line(f"Attached MSFE to encoder (dim={dim_enc}, patches={patch_sizes}, bins={args.msfe_num_bins}, fusion={args.msfe_fusion})", args.log_file)
            if args.msfe_loc in ("decoder", "both"):
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                msfe_dec = MultiScaleFrequencyEnhancement(
                    dim=c_dec,
                    patch_sizes=patch_sizes,
                    num_radial_bins=args.msfe_num_bins,
                    fusion_method=args.msfe_fusion,
                )
                model.mask_decoder.msfe_gate = msfe_dec
                model.mask_decoder.msfe_strength_dec = float(args.msfe_strength_dec)
                log_line(f"Attached MSFE to decoder (dim={c_dec}, patches={patch_sizes}, bins={args.msfe_num_bins}, fusion={args.msfe_fusion})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach MSFE: {e}", args.log_file)
    # Attach Task Tokens (Learnable Prompt Tokens) if requested
    if getattr(args, "use_task_tokens", False):
        try:
            embed_dim = model.prompt_encoder.embed_dim
            num_tokens = args.num_task_tokens
            init_scale = args.task_token_init_scale
            # Create learnable task tokens and attach to prompt_encoder
            task_tokens = torch.nn.Parameter(
                torch.randn(1, num_tokens, embed_dim) * init_scale
            )
            model.prompt_encoder.task_tokens = task_tokens
            log_line(f"Attached Task Tokens to PromptEncoder (num_tokens={num_tokens}, embed_dim={embed_dim}, init_scale={init_scale})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach Task Tokens: {e}", args.log_file)
    model.to(device)

    pgap = None
    if args.use_pgap:
        from efficient_sam.PGAP import PhasePromptGenerator
        pgap = PhasePromptGenerator(
            top_k=args.pgap_top_k,
            input_size=(args.size, args.size),
            min_dist=args.pgap_min_dist,
            saliency_thr=args.pgap_saliency_thr,
            blur_kernel_size=args.pgap_blur_kernel,
            blur_sigma=args.pgap_blur_sigma,
            use_window=not args.pgap_no_window,
            border_width=args.pgap_border_width,
            dynamic_thr=not args.pgap_no_dynamic_thr,
            dynamic_thr_quantile=args.pgap_dyn_quantile,
            dynamic_thr_mode=args.pgap_dyn_mode,
            dynamic_top_k=not args.pgap_no_dynamic_topk,
            min_top_k=args.pgap_min_top_k,
            use_dct=args.pgap_use_dct,
        ).to(device)
        pgap.eval()

    # Initialize FAB Loss (Frequency-Aware Boundary Loss) if requested
    fab_criterion = None
    if getattr(args, "use_fab_loss", False):
        try:
            from efficient_sam.fab_loss import build_fab_loss
            fab_criterion = build_fab_loss(
                num_bins=args.fab_num_bins,
                boundary_width=args.fab_boundary_width,
                high_freq_weight=args.fab_high_freq_weight,
                use_multiscale=True,
            ).to(device)
            log_line(f"Initialized FAB Loss (bins={args.fab_num_bins}, boundary_width={args.fab_boundary_width}, high_freq_weight={args.fab_high_freq_weight})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to initialize FAB Loss: {e}", args.log_file)

    # Initialize SCR Loss (Signal-to-Clutter Ratio Loss) if requested
    scr_criterion = None
    if getattr(args, "use_scr_loss", False):
        try:
            from efficient_sam.scr_loss import build_scr_loss
            scr_criterion = build_scr_loss(
                annular_inner_k=args.scr_inner_k,
                annular_outer_k=args.scr_outer_k,
            ).to(device)
            log_line(f"Initialized SCR Loss (inner_k={args.scr_inner_k}, outer_k={args.scr_outer_k}, weight={args.scr_weight})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to initialize SCR Loss: {e}", args.log_file)

    # Log Prompt-Robust Loss config
    if getattr(args, "use_prompt_robust_loss", False):
        log_line(f"Enabled Prompt-Robust Consistency Loss (weight={args.prompt_robust_weight}, perturb_std={args.prompt_robust_perturb_std})", args.log_file)

    # Stage-1: freeze image encoder
    for p_ in model.image_encoder.parameters():
        p_.requires_grad = False
    if args.use_fs_adapter:
        try:
            from efficient_sam.efficient_sam_encoder_hq import FSAdapter
            fs_tensors = 0
            for m in model.image_encoder.modules():
                if isinstance(m, FSAdapter):
                    for p in m.parameters():
                        p.requires_grad = True
                        fs_tensors += 1
            if fs_tensors > 0:
                log_line("Enabled FSAdapter params during encoder freeze.", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to enable FSAdapter during freeze: {e}", args.log_file)

    # Configure which head params are trainable initially
    # Follow HQ-SAM: only train HQ-specific layers by default
    def mark_maskdecoder_stage1(md):
        for n, p in md.named_parameters():
            p.requires_grad = False
        allow_keys = [
            "hf_token", "hf_mlp", "compress_vit_feat", "embedding_encoder", "embedding_maskfeature",
        ]
        for key in allow_keys:
            mod = getattr(md, key, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True

    if args.freeze_maskdecoder_to_hq:
        mark_maskdecoder_stage1(model.mask_decoder)
    else:
        for p_ in model.mask_decoder.parameters():
            p_.requires_grad = True

    # Prompt encoder trainable during freeze?
    for p_ in model.prompt_encoder.parameters():
        p_.requires_grad = bool(args.train_prompt_encoder_during_freeze)

    # Collect params for optimizer
    head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
    if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
        head_params += list(model.saliency_adapter.parameters())
    if hasattr(model, "ms_aggregator") and model.ms_aggregator is not None:
        head_params += list(model.ms_aggregator.parameters())
    if hasattr(model, "detail_enhancer") and model.detail_enhancer is not None:
        head_params += list(model.detail_enhancer.parameters())

    # MLLM text modules (global CLIP feature -> FiLM / sparse prompt / dense mask prompt)
    text_conditioner = None
    text_sparse_prompt = None
    text_dense_prompt = None
    bifusion_adapter = None
    backbone_bifusion_adapter = None
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    if getattr(args, "use_mllm_prompt", False):
        img_dim = 256  # EfficientSAM-ViTT image embedding dim
        if not pgap_text_prior_only and not getattr(args, "disable_text_conditioner", False):
            text_conditioner = build_text_conditioner(
                img_dim=img_dim,
                text_dim=args.mllm_text_dim,
            ).to(device)
            head_params += list(text_conditioner.parameters())
            n_tc = sum(p.numel() for p in text_conditioner.parameters())
            log_line(f"MLLM TextConditioner enabled: text_dim={args.mllm_text_dim}, params={n_tc}", args.log_file)
        elif getattr(args, "disable_text_conditioner", False):
            log_line("MLLM TextConditioner disabled by --disable_text_conditioner.", args.log_file)
        else:
            log_line("MLLM TextConditioner disabled by --pgap_text_prior_only.", args.log_file)
        if getattr(args, "use_text_sparse_prompt", False) and not pgap_text_prior_only:
            text_sparse_prompt = build_text_sparse_prompt_projector(
                text_dim=args.mllm_text_dim,
                embed_dim=getattr(model.prompt_encoder, "embed_dim", 256),
                num_tokens=max(1, int(args.text_sparse_num_tokens)),
                init_scale=float(args.text_sparse_init_scale),
                use_raw_global_gate=bool(getattr(args, "text_sparse_raw_global_gate", False)),
                raw_global_gate_init_bias=float(getattr(args, "text_sparse_raw_global_gate_init_bias", -2.0)),
            ).to(device)
            head_params += list(text_sparse_prompt.parameters())
            n_tsp = sum(p.numel() for p in text_sparse_prompt.parameters())
            log_line(
                f"Text sparse prompt enabled: source={args.text_sparse_prompt_source}, tokens={args.text_sparse_num_tokens}, gate={bool(getattr(args, 'text_sparse_raw_global_gate', False))}, params={n_tsp}",
                args.log_file,
            )
        elif getattr(args, "use_text_sparse_prompt", False) and pgap_text_prior_only:
            log_line("[info] Ignoring --use_text_sparse_prompt because --pgap_text_prior_only is enabled.", args.log_file)
        if getattr(args, "use_text_dense_prompt", False):
            dense_variant = getattr(args, "text_dense_prompt_type", "global")
            if dense_variant == "token_xattn":
                text_dense_prompt = build_text_dense_mask_prompt_generator_v2(
                    img_dim=img_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                    num_heads=max(1, int(args.text_dense_num_heads)),
                ).to(device)
            else:
                text_dense_prompt = build_text_dense_mask_prompt_generator(
                    img_dim=img_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                ).to(device)
            head_params += list(text_dense_prompt.parameters())
            n_tdp = sum(p.numel() for p in text_dense_prompt.parameters())
            log_line(
                f"Text dense mask prompt enabled: type={dense_variant}, hidden={args.text_dense_hidden_dim}, "
                f"heads={getattr(args, 'text_dense_num_heads', 4)}, alpha={args.text_dense_prompt_merge_alpha}, params={n_tdp}",
                args.log_file,
            )
        if getattr(args, "use_bifusion_adapter", False):
            if pgap_text_prior_only:
                log_line("[info] Disabling BiFusion due to --pgap_text_prior_only.", args.log_file)
            else:
                try:
                    interms_dim = int(model.image_encoder.patch_embed.proj.out_channels)
                except Exception:
                    interms_dim = int(getattr(args, "bifusion_interms_dim", 192))
                bifusion_adapter = build_bifusion_adapter_lite(
                    img_dim=img_dim,
                    interms_dim=interms_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.bifusion_hidden_dim)),
                    num_heads=max(1, int(args.bifusion_num_heads)),
                    use_interms_level=not bool(getattr(args, "bifusion_disable_interms_level", False)),
                    img_res_scale=float(getattr(args, "bifusion_img_res_scale", 1.0)),
                    interms_res_scale=float(getattr(args, "bifusion_interms_res_scale", 1.0)),
                    text_res_scale=float(getattr(args, "bifusion_text_res_scale", 1.0)),
                ).to(device)
                head_params += list(bifusion_adapter.parameters())
                n_bf = sum(p.numel() for p in bifusion_adapter.parameters())
                log_line(
                    f"BiFusion adapter enabled: interms+img levels, hidden={args.bifusion_hidden_dim}, "
                    f"heads={args.bifusion_num_heads}, params={n_bf}",
                    args.log_file,
                )
        use_plain_backbone_bifusion = bool(getattr(args, "use_bifusion_backbone_blocks", False))
        use_gated_backbone_bifusion = bool(getattr(args, "use_gated_bifusion_backbone_blocks", False))
        if use_plain_backbone_bifusion and use_gated_backbone_bifusion:
            log_line("[warn] Both --use_bifusion_backbone_blocks and --use_gated_bifusion_backbone_blocks are set; using gated backbone BiFusion.", args.log_file)
            use_plain_backbone_bifusion = False
        if use_plain_backbone_bifusion or use_gated_backbone_bifusion:
            if pgap_text_prior_only:
                log_line("[info] Disabling backbone BiFusion due to --pgap_text_prior_only.", args.log_file)
            else:
                try:
                    vision_dim = int(model.image_encoder.patch_embed.proj.out_channels)
                except Exception:
                    vision_dim = int(getattr(args, "bifusion_interms_dim", 192))
                num_layers = len(getattr(model.image_encoder, "blocks", []))
                common_kwargs = dict(
                    num_layers=max(1, int(num_layers)),
                    vision_dim=vision_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.bifusion_hidden_dim)),
                    num_heads=max(1, int(args.bifusion_num_heads)),
                    apply_every=max(1, int(getattr(args, "bifusion_block_apply_every", 1))),
                    vision_res_scale=float(getattr(args, "bifusion_block_vision_res_scale", 1.0)),
                    text_res_scale=float(getattr(args, "bifusion_block_text_res_scale", 1.0)),
                )
                if use_gated_backbone_bifusion:
                    backbone_bifusion_adapter = build_gated_backbone_bifusion_block_adapter(
                        gate_hidden_dim=int(getattr(args, "bifusion_gate_hidden_dim", 0)),
                        gate_init_bias=float(getattr(args, "bifusion_gate_init_bias", -2.0)),
                        **common_kwargs,
                    ).to(device)
                else:
                    backbone_bifusion_adapter = build_backbone_bifusion_block_adapter(
                        **common_kwargs,
                    ).to(device)
                head_params += list(backbone_bifusion_adapter.parameters())
                if hasattr(model.image_encoder, "set_text_block_fuser"):
                    model.image_encoder.set_text_block_fuser(backbone_bifusion_adapter)
                else:
                    model.image_encoder.block_text_fuser = backbone_bifusion_adapter
                n_bfb = sum(p.numel() for p in backbone_bifusion_adapter.parameters())
                if use_gated_backbone_bifusion:
                    log_line(
                        f"Gated Backbone BiFusion enabled: layers={num_layers}, hidden={args.bifusion_hidden_dim}, "
                        f"heads={args.bifusion_num_heads}, every={getattr(args, 'bifusion_block_apply_every', 1)}, "
                        f"gate_hidden={getattr(args, 'bifusion_gate_hidden_dim', 0)}, gate_bias={getattr(args, 'bifusion_gate_init_bias', -2.0)}, params={n_bfb}",
                        args.log_file,
                    )
                else:
                    log_line(
                        f"Backbone BiFusion enabled: layers={num_layers}, hidden={args.bifusion_hidden_dim}, "
                        f"heads={args.bifusion_num_heads}, every={getattr(args, 'bifusion_block_apply_every', 1)}, params={n_bfb}",
                        args.log_file,
                    )
    elif (
        getattr(args, "use_text_sparse_prompt", False)
        or getattr(args, "use_text_dense_prompt", False)
        or getattr(args, "use_bifusion_backbone_blocks", False)
        or getattr(args, "use_gated_bifusion_backbone_blocks", False)
    ):
        log_line("[warn] Text sparse/dense/backbone-bifusion flags require --use_mllm_prompt; ignoring.", args.log_file)
    elif getattr(args, "use_bifusion_adapter", False):
        log_line("[warn] --use_bifusion_adapter requires --use_mllm_prompt; ignoring.", args.log_file)
    if getattr(args, "pgap_text_fuse_internal", False):
        if not getattr(args, "use_pgap", False):
            log_line("[warn] --pgap_text_fuse_internal is set but --use_pgap is disabled; internal fusion will not run.", args.log_file)
        if not (getattr(args, "use_mllm_prompt", False) and getattr(args, "use_text_dense_prompt", False)):
            log_line("[warn] --pgap_text_fuse_internal requires --use_mllm_prompt and --use_text_dense_prompt for text prior; falling back to PGAP-only prompts.", args.log_file)
    if getattr(args, "pgap_text_prior_only", False):
        if not getattr(args, "use_pgap", False):
            log_line("[warn] --pgap_text_prior_only has no effect because --use_pgap is disabled.", args.log_file)
        if not getattr(args, "pgap_text_fuse_internal", False):
            log_line("[warn] --pgap_text_prior_only is set without --pgap_text_fuse_internal; text will not affect PGAP prompts.", args.log_file)
    head_params = _dedup_trainable_params(head_params)
    enc_params = _exclude_params(model.image_encoder.parameters(), head_params)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": enc_params, "lr": args.lr_encoder},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = make_scaler(device)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args,
            pgap=pgap, fab_criterion=fab_criterion, scr_criterion=scr_criterion,
            text_conditioner=text_conditioner,
            text_sparse_prompt=text_sparse_prompt,
            text_dense_prompt=text_dense_prompt,
            bifusion_adapter=bifusion_adapter,
            backbone_bifusion_adapter=backbone_bifusion_adapter,
        )
        miou, niou, mf1, pd_val, fa_val, thr_used = validate(
            model, val_loader, device, args, epoch, pgap=pgap,
            text_conditioner=text_conditioner,
            text_sparse_prompt=text_sparse_prompt,
            text_dense_prompt=text_dense_prompt,
            bifusion_adapter=bifusion_adapter,
            backbone_bifusion_adapter=backbone_bifusion_adapter,
        )
        dt = time.time() - t0
        log_line(
            f"[Epoch {epoch:03d}] loss={train_loss:.4f} miou={miou:.4f} niou={niou:.4f} f1={mf1:.4f} "
            f"pd={pd_val:.4f} fa={fa_val:.6f} thr={thr_used:.2f} time={dt:.1f}s",
            args.log_file,
        )

        # Unfreeze schedule
        unfreeze_epoch = (args.epochs // 4) if (args.freeze_encoder_epochs is None or args.freeze_encoder_epochs <= 0) else args.freeze_encoder_epochs
        if epoch == max(1, unfreeze_epoch):
            for p_ in model.image_encoder.parameters():
                p_.requires_grad = True
            if args.unfreeze_all_when_encoder:
                for p_ in model.mask_decoder.parameters():
                    p_.requires_grad = True
                for p_ in model.prompt_encoder.parameters():
                    p_.requires_grad = True
            # Rebuild optimizer to include newly trainable params
            head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
            if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
                head_params += list(model.saliency_adapter.parameters())
            if hasattr(model, "ms_aggregator") and model.ms_aggregator is not None:
                head_params += list(model.ms_aggregator.parameters())
            if hasattr(model, "detail_enhancer") and model.detail_enhancer is not None:
                head_params += list(model.detail_enhancer.parameters())
            if text_conditioner is not None:
                head_params += list(text_conditioner.parameters())
            if text_sparse_prompt is not None:
                head_params += list(text_sparse_prompt.parameters())
            if text_dense_prompt is not None:
                head_params += list(text_dense_prompt.parameters())
            if bifusion_adapter is not None:
                head_params += list(bifusion_adapter.parameters())
            if backbone_bifusion_adapter is not None:
                head_params += list(backbone_bifusion_adapter.parameters())
            head_params = _dedup_trainable_params(head_params)
            enc_params = _exclude_params(model.image_encoder.parameters(), head_params)
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": args.lr_head},
                    {"params": enc_params, "lr": args.lr_encoder},
                ],
                weight_decay=args.weight_decay,
            )
            log_line(
                f"Unfroze at epoch {epoch}: encoder + {'all heads' if args.unfreeze_all_when_encoder else 'keep current head mask' }.",
                args.log_file,
            )

        is_best = miou > best_iou
        if is_best:
            best_iou = miou
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "args": vars(args),
        }
        if text_conditioner is not None:
            ckpt["text_conditioner"] = text_conditioner.state_dict()
        if text_sparse_prompt is not None:
            ckpt["text_sparse_prompt"] = text_sparse_prompt.state_dict()
        if text_dense_prompt is not None:
            ckpt["text_dense_prompt"] = text_dense_prompt.state_dict()
        if bifusion_adapter is not None:
            ckpt["bifusion_adapter"] = bifusion_adapter.state_dict()
        if backbone_bifusion_adapter is not None:
            ckpt["backbone_bifusion_adapter"] = backbone_bifusion_adapter.state_dict()
        if is_best:
            metric_tag = format_metric_tag(epoch, miou, niou, mf1, pd_val, fa_val)
            torch.save(ckpt, os.path.join(args.out_dir, f"best_{metric_tag}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()


