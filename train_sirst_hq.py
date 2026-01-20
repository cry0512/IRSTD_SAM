import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
try:
    from skimage import measure
except Exception:
    measure = None
# Cross-version AMP import
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new
    def autocast_ctx(device: str):
        return _autocast_new("cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    def make_scaler(device: str):
        return _GradScaler_new("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old
    def autocast_ctx(device: str):
        return _autocast_old()
    def make_scaler(device: str):
        return _GradScaler_old()

from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


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


def _build_pgap_prompts(pgap, images, masks, args):
    pgap_pts, pgap_lbl, saliency = pgap(images)
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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, pgap=None):
    model.train()
    if pgap is not None:
        pgap.train()
    bce = nn.BCEWithLogitsLoss()
    meter_loss, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape

        with autocast_ctx(device):
            img_emb, interms = model.get_image_embeddings(images)
            if pgap is not None:
                pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(pgap, images, masks, args)
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
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
            # HQ warmup: force using only HQ mask during early epochs
            use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
            pred_masks, _ = model.predict_masks(
                img_emb,
                interms,
                pts,
                lbl,
                multimask_output=False,
                input_h=H,
                input_w=W,
                output_h=H,
                output_w=W,
                hq_token_only=use_hq_only,
            )
            logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
            loss = bce(logits, masks.unsqueeze(1)) + dice_loss(logits, masks.unsqueeze(1))
            freq_weight = float(getattr(args, "freq_consistency_weight", 0.0))
            if freq_weight > 0.0:
                bins = max(2, int(getattr(args, "freq_consistency_bins", 32)))
                pred_prob = torch.sigmoid(logits)
                gt_mask = masks.unsqueeze(1).float()
                pred_profile = radial_frequency_profile(pred_prob, bins)
                gt_profile = radial_frequency_profile(gt_mask, bins)
                freq_loss = torch.mean(torch.abs(pred_profile - gt_profile))
                loss = loss + freq_weight * freq_loss
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

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter_loss += loss.item() * B
        n += B
    return meter_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, args, epoch: int, pgap=None):
    model.eval()
    if pgap is not None:
        pgap.eval()
    total_inter = 0.0
    total_union = 0.0
    niou_sum = 0.0
    niou_count = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    thr_sum = 0.0
    thr_count = 0
    pd_fa = PD_FA(distance_thresh=getattr(args, "pd_fa_dist", 3))
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape
        img_emb, interms = model.get_image_embeddings(images)
        if pgap is not None:
            if getattr(args, "pgap_two_stage", False):
                pgap_pts, pgap_lbl, saliency = pgap(images)
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
                pos_pts, pos_lbl = _select_topk_points(pgap_pts, pgap_lbl, args.pgap_stage1_top_k)
                pts1 = pos_pts.unsqueeze(1).to(device)
                lbl1 = pos_lbl.unsqueeze(1).to(device)
                use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
                pred_masks1, _ = model.predict_masks(
                    img_emb,
                    interms,
                    pts1,
                    lbl1,
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
                pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(pgap, images, masks, args)
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
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
        use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
        pred_masks, _ = model.predict_masks(
            img_emb,
            interms,
            pts,
            lbl,
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

        prob = torch.sigmoid(logits)
        pred = (prob >= thr_used).float()
        target = masks.unsqueeze(1).float()

        inter = (pred * target).sum().item()
        union = (pred + target - pred * target).sum().item()
        total_inter += inter
        total_union += union

        inter_s = (pred * target).sum(dim=(1, 2, 3))
        union_s = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou_s = torch.where(union_s > 0, inter_s / union_s, torch.ones_like(union_s))
        niou_sum += iou_s.sum().item()
        niou_count += int(iou_s.numel())

        tp += (pred * target).sum().item()
        fp += (pred * (1 - target)).sum().item()
        fn += ((1 - pred) * target).sum().item()

        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        for b in range(pred_cpu.shape[0]):
            pd_fa.update(pred_cpu[b, 0], target_cpu[b, 0], (H, W))

        thr_sum += float(thr_used)
        thr_count += 1

    miou_avg = total_inter / total_union if total_union > 0 else 0.0
    niou_avg = niou_sum / niou_count if niou_count > 0 else 0.0
    denom = (2.0 * tp + fp + fn)
    f1_avg = (2.0 * tp) / denom if denom > 0 else 0.0
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
    p.add_argument("--freq_consistency_weight", type=float, default=0.0,
                   help="Weight for radial frequency consistency loss.")
    p.add_argument("--freq_consistency_bins", type=int, default=32,
                   help="Radial bins used for frequency consistency loss.")
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
    p.add_argument("--boundary_prior_sampling", action="store_true",
                   help="Prefer sampling points near GT boundary.")
    p.add_argument("--boundary_ratio", type=float, default=0.5,
                   help="Fraction of pos/neg points sampled from boundary region.")
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
    p.add_argument("--pgap_label_by_gt", action="store_true",
                   help="Use GT to relabel PGAP points: inside=pos, outside=neg.")
    p.add_argument("--pgap_min_pos", type=int, default=1)
    p.add_argument("--pgap_max_neg", type=int, default=2)
    p.add_argument("--pgap_two_stage", action="store_true",
                   help="Two-stage prompting in validation: pos first, then add negatives outside coarse mask.")
    p.add_argument("--pgap_stage1_top_k", type=int, default=1)
    p.add_argument("--pgap_stage1_thr", type=float, default=0.5)
    p.add_argument("--pgap_stage2_neg", type=int, default=2)
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
    )

    # Model
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=args.init_from_baseline,
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

    # Stage-1: freeze image encoder
    for p_ in model.image_encoder.parameters():
        p_.requires_grad = False

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
    enc_params = [p_ for p_ in model.image_encoder.parameters() if p_.requires_grad]

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
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, pgap=pgap)
        miou, niou, mf1, pd_val, fa_val, thr_used = validate(model, val_loader, device, args, epoch, pgap=pgap)
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
            enc_params = [p_ for p_ in model.image_encoder.parameters() if p_.requires_grad]
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
        if is_best:
            metric_tag = format_metric_tag(epoch, miou, niou, mf1, pd_val, fa_val)
            torch.save(ckpt, os.path.join(args.out_dir, f"best_{metric_tag}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()









