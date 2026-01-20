import os
import time
import argparse
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _boundary_map_from_mask(mask_2d_float: torch.Tensor) -> torch.Tensor:
    m = mask_2d_float.unsqueeze(0).unsqueeze(0)
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    erode = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    b = (dil - erode).clamp(min=0.0, max=1.0)
    return (b[0, 0] > 0).to(mask_2d_float.dtype)


def sample_points_from_mask(
    mask_bhw: torch.Tensor,
    n_pos=4,
    n_neg=4,
    boundary_prior: bool = False,
    boundary_ratio: float = 0.5,
):
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


def point_supervision_loss(logits_b1hw, pts_b1n2, lbl_b1n, pos_weight=1.0):
    # logits_b1hw: [B,1,H,W], pts/lbl: [B,1,N,(2)]
    pts = pts_b1n2[:, 0, ...]
    lbl = lbl_b1n[:, 0, ...]
    valid = lbl >= 0
    if not valid.any():
        return logits_b1hw.sum() * 0.0, logits_b1hw.sum() * 0.0
    B, _, H, W = logits_b1hw.shape
    x = pts[..., 0].round().long().clamp(0, W - 1)
    y = pts[..., 1].round().long().clamp(0, H - 1)
    b_idx = torch.arange(B, device=logits_b1hw.device).view(B, 1)
    logits_pts = logits_b1hw[:, 0][b_idx, y, x]
    targets = lbl.float()
    valid_f = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits_pts, targets, reduction="none")
    if pos_weight != 1.0:
        bce = bce * (targets * (pos_weight - 1.0) + 1.0)
    bce = (bce * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits_pts) * valid_f
    targets = targets * valid_f
    num = 2 * (probs * targets).sum(dim=1) + 1.0
    den = probs.sum(dim=1) + targets.sum(dim=1) + 1.0
    dice = 1 - (num / den)
    dice = dice.mean()
    return bce, dice


def _make_disk_kernel(radius: int, device, dtype):
    r = int(radius)
    if r <= 0:
        return None
    grid = torch.arange(-r, r + 1, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(grid, grid)
    return (xx.pow(2) + yy.pow(2) <= r * r).float()


def _make_gaussian_kernel(radius: int, sigma: float, device, dtype):
    r = int(radius)
    if r <= 0:
        return None
    sigma = max(float(sigma), 1e-6)
    grid = torch.arange(-r, r + 1, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(grid, grid)
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
    return kernel / kernel.max().clamp_min(1e-6)


def _paste_kernel_2d(mask_2d: torch.Tensor, kernel_2d: torch.Tensor, x0: int, y0: int, mode: str = "max"):
    h, w = mask_2d.shape
    r = kernel_2d.shape[0] // 2
    x0 = int(x0)
    y0 = int(y0)
    if x0 < 0 or y0 < 0 or x0 >= w or y0 >= h:
        return
    x1 = x0 - r
    x2 = x0 + r
    y1 = y0 - r
    y2 = y0 + r
    xs0 = max(0, x1)
    xs1 = min(w - 1, x2)
    ys0 = max(0, y1)
    ys1 = min(h - 1, y2)
    kx0 = xs0 - x1
    kx1 = kx0 + (xs1 - xs0) + 1
    ky0 = ys0 - y1
    ky1 = ky0 + (ys1 - ys0) + 1
    patch = mask_2d[ys0:ys1 + 1, xs0:xs1 + 1]
    kpatch = kernel_2d[ky0:ky1, kx0:kx1]
    if mode == "max":
        mask_2d[ys0:ys1 + 1, xs0:xs1 + 1] = torch.maximum(patch, kpatch)
    elif mode == "erase":
        mask_2d[ys0:ys1 + 1, xs0:xs1 + 1] = patch * (1.0 - (kpatch > 0).float())


def build_coarse_mask_from_points(pts_b1n2, lbl_b1n, h: int, w: int,
                                  radius: int, use_gaussian: bool, sigma: float, neg_radius: int):
    device = pts_b1n2.device
    dtype = pts_b1n2.dtype
    bsz = pts_b1n2.shape[0]
    mask = torch.zeros((bsz, 1, h, w), device=device, dtype=torch.float32)
    if radius <= 0:
        return mask
    pos_kernel = _make_gaussian_kernel(radius, sigma, device=device, dtype=dtype) if use_gaussian else _make_disk_kernel(radius, device=device, dtype=dtype)
    if pos_kernel is None:
        return mask
    neg_kernel = _make_disk_kernel(neg_radius, device=device, dtype=dtype) if neg_radius > 0 else None

    pts = pts_b1n2[:, 0, ...]
    lbl = lbl_b1n[:, 0, ...]
    n_pts = pts.shape[1]
    for b in range(bsz):
        for i in range(n_pts):
            if lbl[b, i] != 1:
                continue
            x = int(round(float(pts[b, i, 0])))
            y = int(round(float(pts[b, i, 1])))
            if x < 0 or y < 0:
                continue
            _paste_kernel_2d(mask[b, 0], pos_kernel, x, y, mode="max")
        if neg_kernel is not None:
            for i in range(n_pts):
                if lbl[b, i] != 0:
                    continue
                x = int(round(float(pts[b, i, 0])))
                y = int(round(float(pts[b, i, 1])))
                if x < 0 or y < 0:
                    continue
                _paste_kernel_2d(mask[b, 0], neg_kernel, x, y, mode="erase")
    return mask


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


def write_metrics_csv(path: str, row: dict):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_metrics_jsonl(path: str, row: dict):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_line(message: str, log_path: str = None):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    print(message)


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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    bce_full = nn.BCEWithLogitsLoss()
    meter_loss, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch.get("mask")
        if masks is not None:
            masks = masks.to(device, non_blocking=True)
        B, _, H, W = images.shape

        with autocast_ctx(device):
            img_emb, interms = model.get_image_embeddings(images)
            use_points = False
            pts = batch.get("points")
            lbl = batch.get("point_labels")
            if pts is not None and lbl is not None:
                pts = pts.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                if pts.dim() == 3:
                    pts = pts.unsqueeze(1)
                if lbl.dim() == 2:
                    lbl = lbl.unsqueeze(1)
                use_points = (lbl >= 0).any().item()
            if not use_points:
                if masks is None:
                    raise ValueError("Masks are required when no point annotations are provided.")
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
            bce_pts, dice_pts = point_supervision_loss(logits, pts, lbl, pos_weight=args.point_pos_weight)
            loss = args.point_bce_weight * bce_pts + args.point_dice_weight * dice_pts
            if args.area_reg_weight > 0.0:
                loss = loss + args.area_reg_weight * torch.sigmoid(logits).mean()
            if args.use_coarse_mask_loss:
                if (lbl == 1).any().item():
                    coarse_mask = build_coarse_mask_from_points(
                        pts, lbl, H, W,
                        radius=args.coarse_radius,
                        use_gaussian=args.coarse_use_gaussian,
                        sigma=args.coarse_sigma,
                        neg_radius=args.coarse_neg_radius,
                    )
                    loss = loss + args.coarse_mask_weight * (bce_full(logits, coarse_mask) + dice_loss(logits, coarse_mask))
            if args.use_mask_loss:
                if masks is None:
                    raise ValueError("Masks are required for --use_mask_loss.")
                loss = loss + args.mask_loss_weight * (bce_full(logits, masks.unsqueeze(1)) + dice_loss(logits, masks.unsqueeze(1)))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter_loss += loss.detach().item() * B
        n += B
    return meter_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, args, epoch: int):
    model.eval()
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
        masks = batch.get("mask")
        if masks is not None:
            masks = masks.to(device, non_blocking=True)
        _, _, H, W = images.shape
        img_emb, interms = model.get_image_embeddings(images)
        use_points = False
        pts = batch.get("points")
        lbl = batch.get("point_labels")
        if pts is not None and lbl is not None:
            pts = pts.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            if pts.dim() == 3:
                pts = pts.unsqueeze(1)
            if lbl.dim() == 2:
                lbl = lbl.unsqueeze(1)
            use_points = (lbl >= 0).any().item()
        if not use_points:
            if masks is None:
                raise ValueError("Masks are required when no point annotations are provided.")
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
        if masks is None:
            raise ValueError("Masks are required for validation metrics.")
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
        bsz = pred_cpu.shape[0]
        for b in range(bsz):
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
    p.add_argument("--points_dir", type=str, default="points",
                   help="Folder under data_root with per-image point .txt (name match, line: x y [label]).")
    p.add_argument("--points_normed", action="store_true",
                   help="Point coords are normalized to [0,1] in original image size (origin top-left).")
    p.add_argument("--points_required", type=int, default=1, choices=[0, 1],
                   help="Require point file for each sample (1) or allow missing (0).")
    p.add_argument("--points_max", type=int, default=0,
                   help="Max points per image (0 = use all).")
    p.add_argument("--points_default_label", type=int, default=1,
                   help="Default label for lines without explicit label (1=pos).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n_pos", type=int, default=1)
    p.add_argument("--n_neg", type=int, default=4)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--out_dir", type=str, default="./outputs_sam_sirst_point_hq")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--model", type=str, default="vitt", choices=["vitt", "vits"])
    p.add_argument("--hq_token_only", action="store_true")
    p.add_argument("--hq_warmup_epochs", type=int, default=0,
                   help="If >0, use HQ token only for the first N epochs.")
    p.add_argument("--init_from_baseline", type=str, default=None,
                   help="Optional path to EfficientSAM baseline checkpoint to partially initialize from.")
    p.add_argument("--point_pos_weight", type=float, default=1.0,
                   help="Positive class weight for point BCE.")
    p.add_argument("--point_bce_weight", type=float, default=1.0,
                   help="Weight for point BCE loss.")
    p.add_argument("--point_dice_weight", type=float, default=1.0,
                   help="Weight for point Dice loss.")
    p.add_argument("--use_coarse_mask_loss", action="store_true",
                   help="Use coarse mask loss derived from points (pure point supervision).")
    p.add_argument("--coarse_radius", type=int, default=5,
                   help="Radius (pixels) for coarse mask around positive points.")
    p.add_argument("--coarse_use_gaussian", action="store_true",
                   help="Use Gaussian kernel for coarse mask instead of hard disk.")
    p.add_argument("--coarse_sigma", type=float, default=2.0,
                   help="Sigma for Gaussian coarse mask kernel.")
    p.add_argument("--coarse_neg_radius", type=int, default=0,
                   help="Radius (pixels) to erase around negative points (0 disables).")
    p.add_argument("--coarse_mask_weight", type=float, default=1.0,
                   help="Weight for coarse mask BCE+Dice loss.")
    p.add_argument("--area_reg_weight", type=float, default=0.0,
                   help="Optional area regularizer weight for small-target bias.")
    p.add_argument("--use_mask_loss", action="store_true",
                   help="Optional: add full-mask BCE+Dice loss (debug/ablation).")
    p.add_argument("--mask_loss_weight", type=float, default=1.0)
    p.add_argument("--boundary_prior_sampling", action="store_true",
                   help="Prefer sampling points near GT boundary.")
    p.add_argument("--boundary_ratio", type=float, default=0.5,
                   help="Fraction of pos/neg points sampled from boundary region.")
    p.add_argument("--val_thr_search", action="store_true",
                   help="Enable validation threshold grid search.")
    p.add_argument("--val_thr_min", type=float, default=0.35)
    p.add_argument("--val_thr_max", type=float, default=0.55)
    p.add_argument("--val_thr_step", type=float, default=0.05)
    p.add_argument("--pd_fa_dist", type=int, default=3,
                   help="Distance threshold (pixels) for PD/FA matching.")
    p.add_argument("--metrics_csv", type=str, default=None,
                   help="Path to CSV metrics log (default: <out_dir>/metrics.csv).")
    p.add_argument("--metrics_json", type=str, default=None,
                   help="Path to JSONL metrics log (default: <out_dir>/metrics.jsonl).")
    p.add_argument("--log_file", type=str, default=None,
                   help="Path to text log file (default: <out_dir>/log.txt).")
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
    if args.metrics_csv is None:
        args.metrics_csv = os.path.join(run_dir, "metrics.csv")
    if args.metrics_json is None:
        args.metrics_json = os.path.join(run_dir, "metrics.jsonl")
    if args.log_file is None:
        args.log_file = os.path.join(run_dir, "log.txt")
    args.out_dir = run_dir
    log_line(f"Run directory: {args.out_dir}", args.log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        points_dir=args.points_dir if args.points_dir else None,
        points_normed=args.points_normed,
        points_default_label=args.points_default_label,
        points_required=bool(args.points_required),
        points_max=args.points_max,
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
        points_dir=args.points_dir if args.points_dir else None,
        points_normed=args.points_normed,
        points_default_label=args.points_default_label,
        points_required=bool(args.points_required),
        points_max=args.points_max,
    )

    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=args.init_from_baseline,
    )
    if args.use_radial_gate_hq:
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

    for p_ in model.image_encoder.parameters():
        p_.requires_grad = False

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

    for p_ in model.prompt_encoder.parameters():
        p_.requires_grad = bool(args.train_prompt_encoder_during_freeze)

    head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
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
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        miou, niou, mf1, pd_val, fa_val, thr_used = validate(model, val_loader, device, args, epoch)
        dt = time.time() - t0
        log_line(
            f"[Epoch {epoch:03d}] loss={train_loss:.4f} miou={miou:.4f} niou={niou:.4f} f1={mf1:.4f} "
            f"pd={pd_val:.4f} fa={fa_val:.6f} thr={thr_used:.2f} time={dt:.1f}s",
            args.log_file,
        )
        metrics_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "miou": float(miou),
            "niou": float(niou),
            "f1": float(mf1),
            "pd": float(pd_val),
            "fa": float(fa_val),
            "thr": float(thr_used),
            "time_sec": float(dt),
        }
        write_metrics_csv(args.metrics_csv, metrics_row)
        write_metrics_jsonl(args.metrics_json, metrics_row)

        unfreeze_epoch = (args.epochs // 4) if (args.freeze_encoder_epochs is None or args.freeze_encoder_epochs <= 0) else args.freeze_encoder_epochs
        if epoch == max(1, unfreeze_epoch):
            for p_ in model.image_encoder.parameters():
                p_.requires_grad = True
            if args.unfreeze_all_when_encoder:
                for p_ in model.mask_decoder.parameters():
                    p_.requires_grad = True
                for p_ in model.prompt_encoder.parameters():
                    p_.requires_grad = True
            head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
            enc_params = [p_ for p_ in model.image_encoder.parameters() if p_.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": args.lr_head},
                    {"params": enc_params, "lr": args.lr_encoder},
                ],
                weight_decay=args.weight_decay,
            )
            log_line(
                f"Unfroze at epoch {epoch}: encoder + {'all heads' if args.unfreeze_all_when_encoder else 'keep current head mask'}.",
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
