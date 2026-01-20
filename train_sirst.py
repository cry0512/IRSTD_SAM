import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
try:
    from skimage import measure
except Exception:
    measure = None
# Cross-version AMP import (PyTorch>=2.0 uses torch.amp, older uses torch.cuda.amp)
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new  # type: ignore
    def autocast_ctx(device: str):
        return _autocast_new('cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    def make_scaler(device: str):
        return _GradScaler_new('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old  # type: ignore
    def autocast_ctx(device: str):
        return _autocast_old()
    def make_scaler(device: str):
        return _GradScaler_old()

from sirst_dataset import make_loader
from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)

from efficient_sam.freq_modules import FreqGate, RadialFreqGate
from efficient_sam.asg import AnisotropicSpectralGating, AnisotropicSpectralGating2
from efficient_sam.PGAP import PhasePromptGenerator

def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


def sample_points_from_mask(mask_bhw: torch.Tensor, n_pos=4, n_neg=4):
    B, H, W = mask_bhw.shape
    device = mask_bhw.device
    pts, labels = [], []
    for b in range(B):
        pos_idx = (mask_bhw[b] > 0).nonzero(as_tuple=False)
        neg_idx = (mask_bhw[b] == 0).nonzero(as_tuple=False)
        npos = min(n_pos, len(pos_idx)) if len(pos_idx) > 0 else 0
        nneg = min(n_neg, len(neg_idx)) if len(neg_idx) > 0 else 0
        pos = (
            pos_idx[torch.randint(len(pos_idx), (npos,), device=pos_idx.device)] if npos > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
        )
        neg = (
            neg_idx[torch.randint(len(neg_idx), (nneg,), device=neg_idx.device)] if nneg > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
        )
        p = torch.cat([pos, neg], dim=0)
        l = torch.cat([torch.ones(npos), torch.zeros(nneg)], dim=0)
        if p.numel() == 0:
            if len(neg_idx) > 0:
                p = neg_idx[torch.randint(len(neg_idx), (n_neg,), device=neg_idx.device)]
                l = torch.zeros(p.shape[0], device=device)
            else:
                p = torch.zeros((1, 2), dtype=torch.long, device=device)
                l = torch.zeros(1, device=device)
        xy = torch.stack([p[:, 1], p[:, 0]], dim=-1).float()  # (x,y)
        pts.append(xy[None, ...])  # 1 query
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


def log_line(message: str, log_path: str = None):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    print(message, flush=True)


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
            img_emb = model.get_image_embeddings(images)
            if pgap is not None:
                pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(pgap, images, masks, args)
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
                pts = pgap_pts.unsqueeze(1)
                lbl = pgap_lbl.unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
            else:
                pts, lbl = sample_points_from_mask(masks, n_pos=args.n_pos, n_neg=args.n_neg)
                pts, lbl = pts.to(device), lbl.to(device)
            pred_masks, _ = model.predict_masks(
                img_emb, pts, lbl, multimask_output=False,
                input_h=H, input_w=W, output_h=H, output_w=W,
            )
            logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
            loss = bce(logits, masks.unsqueeze(1)) + dice_loss(logits, masks.unsqueeze(1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter_loss += loss.item() * B
        n += B
    return meter_loss / max(n, 1)


def validate(model, loader, device, args, pgap=None):
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
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            B, H, W = masks.shape
            img_emb = model.get_image_embeddings(images)
            if pgap is not None:
                if getattr(args, "pgap_two_stage", False):
                    pgap_pts, pgap_lbl, saliency = pgap(images)
                    img_emb = model.apply_saliency_modulation(img_emb, saliency)
                    pos_pts, pos_lbl = _select_topk_points(pgap_pts, pgap_lbl, args.pgap_stage1_top_k)
                    pts1 = pos_pts.unsqueeze(1).to(device)
                    lbl1 = pos_lbl.unsqueeze(1).to(device)
                    pred_masks1, _ = model.predict_masks(
                        img_emb, pts1, lbl1, multimask_output=False,
                        input_h=H, input_w=W, output_h=H, output_w=W,
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
                pts, lbl = sample_points_from_mask(masks, n_pos=args.n_pos, n_neg=args.n_neg)
                pts, lbl = pts.to(device), lbl.to(device)
            pred_masks, _ = model.predict_masks(
                img_emb, pts, lbl, multimask_output=False,
                input_h=H, input_w=W, output_h=H, output_w=W,
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
            for b in range(B):
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
    p.add_argument("--out_dir", type=str, default="./outputs_sam_sirst")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Optional experiment name. If not set, a name is auto-generated from key args + timestamp.")
    p.add_argument("--mask_suffix", type=str, default="",
                   help="Optional suffix for mask filenames before extension, e.g. '_pixels0'.")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--model", type=str, default="vitt", choices=["vitt", "vits"])
    p.add_argument("--val_thr_search", action="store_true",
                   help="Enable validation threshold grid search.")
    p.add_argument("--val_thr_min", type=float, default=0.35)
    p.add_argument("--val_thr_max", type=float, default=0.55)
    p.add_argument("--val_thr_step", type=float, default=0.05)
    p.add_argument("--pd_fa_dist", type=int, default=3,
                   help="Distance threshold for PD/FA metrics (in pixels).")
    p.add_argument("--log_file", type=str, default=None,
                   help="Path to log file (default: <out_dir>/log.txt).")

    # Frequency modules
    p.add_argument("--use_ffc", action="store_true")
    p.add_argument("--ffc_kernel", type=int, default=1)
    p.add_argument("--ffc_use_only_freq", action="store_true")
    p.add_argument("--ffc_fft_norm", type=str, default="ortho")
    p.add_argument("--use_fftformer", action="store_true")
    p.add_argument("--fftf_expansion", type=float, default=3.0)
    p.add_argument("--fftf_patch_size", type=int, default=8)

    p.add_argument("--use_freq_gate", action="store_true")
    p.add_argument("--use_radial_gate", action="store_true")
    p.add_argument("--freq_patch_size", type=int, default=8)
    p.add_argument("--radial_bins", type=int, default=6)
    p.add_argument("--radial_channel_shared", action="store_true")
    # ASG (anisotropic spectral gating)
    p.add_argument("--use_asg", action="store_true", help="Enable AnisotropicSpectralGating in image encoder.")
    p.add_argument("--asg_variant", type=str, default="asg1", choices=["asg1", "asg2"])
    p.add_argument("--asg_radial_bins", type=int, default=64)
    p.add_argument("--asg_angular_bins", type=int, default=128)
    p.add_argument("--asg_strength", type=float, default=1.0)
    # PGAP (phase prompt generator)
    p.add_argument("--use_pgap", action="store_true", help="Use PhasePromptGenerator to auto-generate prompt points.")
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
    args = p.parse_args()

    # Build per-run directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.exp_name is None:
        base = [
            f"model-{args.model}", f"size-{args.size}", f"bs-{args.batch_size}",
            f"lrh-{args.lr_head}", f"lre-{args.lr_encoder}", f"kp-{int(args.keep_ratio_pad)}",
            f"thr-{args.thr}", f"npos-{args.n_pos}", f"nneg-{args.n_neg}",
        ]
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
        args.data_root, args.train_txt,
        batch_size=args.batch_size, size=args.size, augment=True,
        keep_ratio_pad=args.keep_ratio_pad, workers=args.workers,
        shuffle=True, mask_suffix=args.mask_suffix,
    )
    val_loader = make_loader(
        args.data_root, args.val_txt,
        batch_size=max(1, args.batch_size // 2), size=args.size, augment=False,
        keep_ratio_pad=args.keep_ratio_pad, workers=args.workers,
        shuffle=False, mask_suffix=args.mask_suffix,
    )

    # Model
    model = build_efficient_sam_vits() if args.model == "vits" else build_efficient_sam_vitt()

    # Attach FFC (optional)
    if args.use_ffc:
        try:
            dim = model.image_encoder.neck[0].out_channels
        except Exception:
            dim = 256
        from efficient_sam.freq_modules import SpectralTransformLite
        model.image_encoder.ffc = SpectralTransformLite(dim, fu_kernel=args.ffc_kernel, use_only_freq=args.ffc_use_only_freq, fft_norm=args.ffc_fft_norm)

    # Attach frequency gates (optional)
    if args.use_freq_gate or args.use_radial_gate:
        try:
            dim = model.image_encoder.neck[0].out_channels
        except Exception:
            dim = 256
        if args.use_freq_gate:
            model.image_encoder.freq_gate = FreqGate(dim, patch_size=args.freq_patch_size)
        if args.use_radial_gate and not args.use_asg:
            model.image_encoder.radial_gate = RadialFreqGate(dim, patch_size=args.freq_patch_size, num_bins=args.radial_bins, channel_shared=args.radial_channel_shared)

    # Attach ASG (optional)
    if args.use_asg:
        if args.use_radial_gate:
            log_line("[warn] Both ASG and RadialFreqGate are enabled; using ASG and skipping RadialFreqGate.", args.log_file)
        try:
            class _ASGDelta(nn.Module):
                # Convert ASG residual output to a delta for x + gate(x) usage.
                def __init__(self, asg, strength):
                    super().__init__()
                    self.asg = asg
                    self.strength = float(strength)

                def forward(self, x):
                    return self.strength * (self.asg(x) - x)

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
            try:
                dim = model.image_encoder.neck[0].out_channels
            except Exception:
                dim = 256
            asg = asg_cls(dim, enc_hw, enc_hw, **asg_kwargs)
            model.image_encoder.radial_gate = _ASGDelta(asg, args.asg_strength)
        except Exception as e:
            log_line(f"[warn] Failed to attach ASG: {e}", args.log_file)

    model.train()
    model.to(device)

    pgap = None
    if args.use_pgap:
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

    # Freeze encoder; enable grads for attached modules
    for p_ in model.image_encoder.parameters():
        p_.requires_grad = False
    if hasattr(model.image_encoder, "ffc") and model.image_encoder.ffc is not None:
        for p_ in model.image_encoder.ffc.parameters():
            p_.requires_grad = True
    if hasattr(model.image_encoder, "freq_gate") and model.image_encoder.freq_gate is not None:
        for p_ in model.image_encoder.freq_gate.parameters():
            p_.requires_grad = True
    if hasattr(model.image_encoder, "radial_gate") and model.image_encoder.radial_gate is not None:
        for p_ in model.image_encoder.radial_gate.parameters():
            p_.requires_grad = True

    # Optimizer param groups
    head_params = list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
        head_params += list(model.saliency_adapter.parameters())
    if hasattr(model.image_encoder, "freq_gate") and model.image_encoder.freq_gate is not None:
        head_params += list(model.image_encoder.freq_gate.parameters())
    if hasattr(model.image_encoder, "radial_gate") and model.image_encoder.radial_gate is not None:
        head_params += list(model.image_encoder.radial_gate.parameters())
    if hasattr(model.image_encoder, "ffc") and model.image_encoder.ffc is not None:
        head_params += list(model.image_encoder.ffc.parameters())
    if hasattr(model.image_encoder, "fftf") and model.image_encoder.fftf is not None:
        head_params += list(model.image_encoder.fftf.parameters())

    _enc_all = list(model.image_encoder.parameters())
    _exclude = set()
    if hasattr(model.image_encoder, "freq_gate") and model.image_encoder.freq_gate is not None:
        _exclude.update(id(p) for p in model.image_encoder.freq_gate.parameters())
    if hasattr(model.image_encoder, "radial_gate") and model.image_encoder.radial_gate is not None:
        _exclude.update(id(p) for p in model.image_encoder.radial_gate.parameters())
    if hasattr(model.image_encoder, "ffc") and model.image_encoder.ffc is not None:
        _exclude.update(id(p) for p in model.image_encoder.ffc.parameters())
    if hasattr(model.image_encoder, "fftf") and model.image_encoder.fftf is not None:
        _exclude.update(id(p) for p in model.image_encoder.fftf.parameters())
    enc_params = [p for p in _enc_all if id(p) not in _exclude]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": enc_params, "lr": args.lr_encoder},
    ], weight_decay=args.weight_decay)
    scaler = make_scaler(device)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, pgap=pgap)
        miou, niou, mf1, pd_val, fa_val, thr_used = validate(model, val_loader, device, args, pgap=pgap)
        dt = time.time() - t0
        log_line(
            f"[Epoch {epoch:03d}] loss={train_loss:.4f} miou={miou:.4f} niou={niou:.4f} f1={mf1:.4f} "
            f"pd={pd_val:.4f} fa={fa_val:.6f} thr={thr_used:.2f} time={dt:.1f}s",
            args.log_file,
        )

        if epoch == max(3, args.epochs // 4):
            for p_ in model.image_encoder.parameters():
                p_.requires_grad = True
            log_line("Unfroze image encoder for joint fine-tuning.", args.log_file)

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
            metric_tag = format_metric_tag(epoch, miou, niou, mf1, pd_val, fa_val, thr=thr_used)
            torch.save(ckpt, os.path.join(args.out_dir, f"best_{metric_tag}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()


