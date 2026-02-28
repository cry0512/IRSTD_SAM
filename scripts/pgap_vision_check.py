import os
import sys
import argparse
import csv
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sirst_dataset import SIRSTDataset
from efficient_sam.PGAP import PhasePromptGenerator
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_text_conditioner,
    build_text_dense_mask_prompt_generator,
    build_text_dense_mask_prompt_generator_v2,
)


def _to_numpy_gt(gt_t):
    if isinstance(gt_t, torch.Tensor):
        gt = gt_t.detach().cpu().numpy()
    else:
        gt = gt_t
    return (gt > 0).astype(np.uint8)


def _to_base_image(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy()
    if img.ndim == 3:
        base = img[0]
    else:
        base = img
    base = base.astype(np.float32)
    base = base - base.min()
    base = base / (base.max() + 1e-6)
    return base


def _points_to_xy_on_target(
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = point_coords.detach().cpu().numpy()
    lbl = point_labels.detach().cpu().numpy()
    H, W = gt.shape[-2], gt.shape[-1]
    valid = lbl >= 0
    pts = pts[valid]
    lbl = lbl[valid]
    if pts.size == 0:
        xs = np.array([], dtype=int)
        ys = np.array([], dtype=int)
        on_target = np.array([], dtype=bool)
    else:
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        on_target = gt[ys, xs] > 0
    return xs, ys, on_target, lbl


def _plot_points(ax, xs, ys, on_target, title: str, show_legend: bool = False):
    ax.scatter(xs[on_target], ys[on_target], s=35, marker="o", c="lime", label="pts on GT")
    ax.scatter(xs[~on_target], ys[~on_target], s=35, marker="x", c="red", label="pts on BG")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")


@torch.no_grad()
def _run_pgap(img_t: torch.Tensor, pgap: PhasePromptGenerator):
    point_coords, point_labels, saliency = pgap(img_t.unsqueeze(0))
    return point_coords[0], point_labels[0], saliency[0, 0]


@torch.no_grad()
def _build_text_prior(
    img_t: torch.Tensor,
    sample: dict,
    model,
    text_conditioner,
    text_dense_prompt,
    device: str,
    scale: float = 1.0,
) -> Optional[torch.Tensor]:
    if model is None or text_dense_prompt is None:
        return None
    if "clip_text_feat" not in sample and "clip_text_token_feat" not in sample:
        return None

    images = img_t.unsqueeze(0).to(device)
    img_emb, _ = model.get_image_embeddings(images)

    clip_feat = sample.get("clip_text_feat", None)
    if clip_feat is not None:
        clip_feat = clip_feat.unsqueeze(0).to(device)
    clip_token_feat = sample.get("clip_text_token_feat", None)
    clip_token_mask = sample.get("clip_text_attn_mask", None)
    if clip_token_feat is not None:
        clip_token_feat = clip_token_feat.unsqueeze(0).to(device)
    if clip_token_mask is not None:
        clip_token_mask = clip_token_mask.unsqueeze(0).to(device)

    if text_conditioner is not None and clip_feat is not None:
        img_emb = text_conditioner(img_emb, clip_feat)

    out_h, out_w = img_t.shape[-2], img_t.shape[-1]
    if getattr(text_dense_prompt, "expects_token_level", False):
        dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
        if dense_text_input is None:
            return None
        dense = text_dense_prompt(
            img_emb,
            dense_text_input,
            attention_mask=clip_token_mask if clip_token_feat is not None else None,
            output_size=(out_h, out_w),
        )
    else:
        if clip_feat is None:
            return None
        dense = text_dense_prompt(
            img_emb,
            clip_feat,
            output_size=(out_h, out_w),
        )
    dense = dense * float(scale)
    return dense[0, 0].detach().cpu()


@torch.no_grad()
def _rerank_points_with_text(
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    saliency_map: torch.Tensor,
    text_prior: Optional[torch.Tensor],
    alpha: float = 0.5,
    backup_ratio: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if text_prior is None:
        return point_coords, point_labels

    coords = point_coords.clone()
    labels = point_labels.clone()
    valid = labels >= 0
    if valid.sum().item() <= 0:
        return coords, labels

    coords_v = coords[valid]
    labels_v = labels[valid]
    # Keep maps on the same device as point indices for indexing.
    map_device = coords_v.device
    saliency_map = saliency_map.to(map_device)
    text_prior = text_prior.to(map_device)
    h, w = saliency_map.shape[-2], saliency_map.shape[-1]
    xs = coords_v[:, 0].round().long().clamp(0, w - 1)
    ys = coords_v[:, 1].round().long().clamp(0, h - 1)

    s_pgap = saliency_map[ys, xs].float()
    s_text = text_prior[ys, xs].float()
    a = max(0.0, float(alpha))
    s_fused = s_pgap * (1.0 + a * s_text)

    k_total = coords_v.shape[0]
    k_backup = int(round(k_total * max(0.0, min(1.0, float(backup_ratio)))))
    k_backup = min(k_total, max(0, k_backup))
    k_fused = max(0, k_total - k_backup)

    order = []
    used = torch.zeros(k_total, dtype=torch.bool)
    if k_fused > 0:
        fused_idx = torch.topk(s_fused, k=k_fused).indices
        for idx in fused_idx.tolist():
            if not used[idx]:
                used[idx] = True
                order.append(idx)
    if k_backup > 0:
        pgap_idx = torch.topk(s_pgap, k=min(k_total, s_pgap.numel())).indices
        for idx in pgap_idx.tolist():
            if not used[idx]:
                used[idx] = True
                order.append(idx)
            if len(order) >= k_total:
                break

    if len(order) < k_total:
        for idx in range(k_total):
            if not used[idx]:
                order.append(idx)
    order_t = torch.as_tensor(order[:k_total], device=coords_v.device, dtype=torch.long)

    coords_out = coords_v[order_t]
    labels_out = labels_v[order_t]
    coords[valid] = coords_out
    labels[valid] = labels_out
    return coords, labels


def _normalize_map01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    minv = x.min()
    maxv = x.max()
    return (x - minv) / (maxv - minv + 1e-6)


@torch.no_grad()
def _refind_points_with_text(
    pgap: PhasePromptGenerator,
    saliency_map: torch.Tensor,
    text_prior: Optional[torch.Tensor],
    alpha: float = 0.5,
    fuse_mode: str = "mul",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if text_prior is None:
        coords, labels = pgap.extract_points(saliency_map.unsqueeze(0).unsqueeze(0))
        return coords[0], labels[0], saliency_map

    device = saliency_map.device
    s = _normalize_map01(saliency_map)
    t = text_prior.to(device)
    if tuple(t.shape[-2:]) != tuple(s.shape[-2:]):
        t = F.interpolate(
            t.unsqueeze(0).unsqueeze(0),
            size=tuple(s.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    t = _normalize_map01(t)
    a = max(0.0, float(alpha))
    if fuse_mode == "add":
        fused = (1.0 - a) * s + a * t
    else:
        fused = s * (1.0 + a * t)
    fused = _normalize_map01(fused)
    coords, labels = pgap.extract_points(fused.unsqueeze(0).unsqueeze(0))
    return coords[0], labels[0], fused


@torch.no_grad()
def vis_pgap_compare_on_sample(
    img_t,
    gt_t,
    pgap: PhasePromptGenerator,
    save_path=None,
    title="",
    show=False,
    compare_map: Optional[torch.Tensor] = None,
    compare_map_title: str = "Text prior + reranked points",
    fused_point_coords: Optional[torch.Tensor] = None,
    fused_point_labels: Optional[torch.Tensor] = None,
):
    gt = _to_numpy_gt(gt_t)
    base = _to_base_image(img_t)

    point_coords, point_labels, saliency = _run_pgap(img_t, pgap)
    sal = saliency.detach().cpu().numpy()
    xs, ys, on_target, _ = _points_to_xy_on_target(point_coords, point_labels, gt)
    pgap_on = int(on_target.sum())
    pgap_total = int(len(on_target))

    has_text_compare = (
        compare_map is not None and fused_point_coords is not None and fused_point_labels is not None
    )
    if has_text_compare:
        xs_f, ys_f, on_target_f, _ = _points_to_xy_on_target(fused_point_coords, fused_point_labels, gt)
        fused_on = int(on_target_f.sum())
        fused_total = int(len(on_target_f))
    else:
        xs_f = ys_f = on_target_f = None
        fused_on = fused_total = 0

    if has_text_compare:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes = list(axes)

    axes[0].imshow(base, cmap="gray")
    _plot_points(axes[0], xs, ys, on_target, "Image + PGAP points", show_legend=True)

    axes[1].imshow(sal, cmap="hot")
    _plot_points(axes[1], xs, ys, on_target, "PGAP saliency + points")

    if has_text_compare:
        cmp_np = compare_map.detach().cpu().numpy()
        axes[2].imshow(cmp_np, cmap="viridis")
        _plot_points(axes[2], xs_f, ys_f, on_target_f, compare_map_title)

        axes[3].imshow(base, cmap="gray")
        _plot_points(
            axes[3],
            xs_f,
            ys_f,
            on_target_f,
            f"Image + PGAP+Text ({fused_on}/{max(1, fused_total)})",
        )

        axes[4].imshow(base, cmap="gray")
        axes[4].contour(gt.astype(float), levels=[0.5], linewidths=1, colors="yellow")
        axes[4].scatter(xs, ys, s=22, marker="x", c="cyan", label="PGAP")
        axes[4].scatter(xs_f, ys_f, s=22, marker="o", facecolors="none", edgecolors="lime", label="PGAP+Text")
        axes[4].set_title("GT + point compare")
        axes[4].legend(loc="lower right", fontsize=8)
        axes[4].axis("off")
    else:
        axes[2].imshow(base, cmap="gray")
        axes[2].contour(gt.astype(float), levels=[0.5], linewidths=1)
        axes[2].set_title("GT contour")
        axes[2].axis("off")

    if title:
        if has_text_compare:
            fig.suptitle(f"{title} | PGAP {pgap_on}/{pgap_total} | PGAP+Text {fused_on}/{fused_total}")
        else:
            fig.suptitle(f"{title} | PGAP {pgap_on}/{pgap_total}")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return {
        "pgap_on": pgap_on,
        "pgap_total": pgap_total,
        "fused_on": fused_on if has_text_compare else None,
        "fused_total": fused_total if has_text_compare else None,
    }


def _load_ckpt_text_modules(args, device):
    if not args.text_ckpt:
        return None, None, None, None

    if not os.path.isfile(args.text_ckpt):
        print(f"[warn] text_ckpt not found: {args.text_ckpt}")
        return None, None, None, None

    ckpt = torch.load(args.text_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        print(f"[warn] Unsupported checkpoint format: {type(ckpt)}")
        return None, None, None, None

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=ckpt_args.get("init_from_baseline", None),
        use_adapter=bool(ckpt_args.get("use_fs_adapter", True)),
        use_ms_fusion=bool(ckpt_args.get("use_ms_fusion", False)),
        use_detail_enhancer=bool(ckpt_args.get("use_detail_enhancer", False)),
        early_exit_layer=ckpt_args.get("early_exit_layer", None),
    ).to(device)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[info] model.load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[warn] Checkpoint has no 'model' state_dict; text prior visualization may be unavailable.")
        model = None

    text_dim = int(ckpt_args.get("mllm_text_dim", args.mllm_text_dim))
    text_conditioner = None
    if "text_conditioner" in ckpt and model is not None:
        text_conditioner = build_text_conditioner(img_dim=256, text_dim=text_dim).to(device)
        miss_tc, unexp_tc = text_conditioner.load_state_dict(ckpt["text_conditioner"], strict=False)
        if miss_tc or unexp_tc:
            print(f"[info] text_conditioner.load_state_dict(strict=False): missing={len(miss_tc)}, unexpected={len(unexp_tc)}")
        text_conditioner.eval()

    text_dense_prompt = None
    if "text_dense_prompt" in ckpt and model is not None:
        variant = args.text_dense_prompt_type
        if variant == "auto":
            variant = str(ckpt_args.get("text_dense_prompt_type", "global"))
        hidden_dim = int(args.text_dense_hidden_dim if args.text_dense_hidden_dim > 0 else ckpt_args.get("text_dense_hidden_dim", 128))
        num_heads = int(args.text_dense_num_heads if args.text_dense_num_heads > 0 else ckpt_args.get("text_dense_num_heads", 4))
        if variant == "token_xattn":
            text_dense_prompt = build_text_dense_mask_prompt_generator_v2(
                img_dim=256,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            ).to(device)
        else:
            text_dense_prompt = build_text_dense_mask_prompt_generator(
                img_dim=256,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        miss_tdp, unexp_tdp = text_dense_prompt.load_state_dict(ckpt["text_dense_prompt"], strict=False)
        if miss_tdp or unexp_tdp:
            print(f"[info] text_dense_prompt.load_state_dict(strict=False): missing={len(miss_tdp)}, unexpected={len(unexp_tdp)}")
        text_dense_prompt.eval()
        print(f"[info] Loaded text_dense_prompt for visualization: type={variant}, hidden={hidden_dim}, heads={num_heads}")
    else:
        print("[warn] Checkpoint does not contain 'text_dense_prompt'; text rerank visualization disabled.")

    if model is not None:
        model.eval()
    return model, text_conditioner, text_dense_prompt, ckpt_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_txt", type=str, default="50_50/test.txt")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--keep_ratio_pad", action="store_true")
    parser.add_argument("--mask_suffix", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./outputs_pgap_vis")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sctransnet_preproc", action="store_true")
    parser.add_argument("--sc_pos_prob", type=float, default=0.5)
    parser.add_argument("--sc_dataset_name", type=str, default=None)
    parser.add_argument("--mllm_features_path", type=str, default=None,
                        help="Optional MLLM CLIP features (.pt); supports global or token-level files.")
    parser.add_argument("--text_ckpt", type=str, default=None,
                        help="Training checkpoint (best.pt) containing text_dense_prompt weights for text-aware visualization.")
    parser.add_argument("--mllm_text_dim", type=int, default=512)
    parser.add_argument("--text_dense_prompt_type", type=str, default="auto",
                        choices=["auto", "global", "token_xattn"])
    parser.add_argument("--text_dense_hidden_dim", type=int, default=-1)
    parser.add_argument("--text_dense_num_heads", type=int, default=-1)
    parser.add_argument("--text_dense_prompt_scale", type=float, default=1.0)
    parser.add_argument("--text_rerank_alpha", type=float, default=0.5,
                        help="Weight for text prior in PGAP point reranking: s = s_pgap * (1 + alpha*s_text).")
    parser.add_argument("--text_rerank_backup_ratio", type=float, default=0.4,
                        help="Fraction of points kept from original PGAP ranking as backup against wrong text.")
    parser.add_argument("--text_point_mode", type=str, default="rerank",
                        choices=["rerank", "refind"],
                        help="rerank: reorder existing PGAP points; refind: fuse PGAP saliency + text prior and re-extract points.")
    parser.add_argument("--text_saliency_fuse_mode", type=str, default="mul",
                        choices=["mul", "add"],
                        help="How to fuse PGAP saliency and text prior when --text_point_mode=refind.")
    parser.add_argument("--pgap_top_k", type=int, default=5)
    parser.add_argument("--pgap_min_dist", type=int, default=10)
    parser.add_argument("--pgap_saliency_thr", type=float, default=0.1)
    parser.add_argument("--pgap_blur_kernel", type=int, default=5)
    parser.add_argument("--pgap_blur_sigma", type=float, default=1.0)
    parser.add_argument("--pgap_border_width", type=int, default=12)
    parser.add_argument("--pgap_no_window", action="store_true")
    parser.add_argument("--pgap_no_dynamic_thr", action="store_true")
    parser.add_argument("--pgap_dyn_quantile", type=float, default=0.9)
    parser.add_argument("--pgap_dyn_mode", type=str, default="max", choices=["max", "replace"])
    parser.add_argument("--pgap_no_dynamic_topk", action="store_true")
    parser.add_argument("--pgap_min_top_k", type=int, default=1)
    parser.add_argument("--pgap_use_dct", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SIRSTDataset(
        root=args.data_root,
        split_txt=args.split_txt,
        size=args.size,
        keep_ratio_pad=args.keep_ratio_pad,
        augment=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=args.sctransnet_preproc,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_pos_prob=args.sc_pos_prob,
        sc_dataset_name=args.sc_dataset_name,
        mllm_features_path=args.mllm_features_path,
    )

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(ds))
    if args.max_samples > 0 and args.max_samples < len(ds):
        indices = rng.choice(indices, size=args.max_samples, replace=False)

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

    model_vis, text_conditioner_vis, text_dense_prompt_vis, _ = _load_ckpt_text_modules(args, device)
    use_text_compare = model_vis is not None and text_dense_prompt_vis is not None and bool(args.mllm_features_path)
    if args.text_ckpt and not args.mllm_features_path:
        print("[warn] --text_ckpt provided but --mllm_features_path missing; cannot build text priors.")
    if args.mllm_features_path and not os.path.isfile(args.mllm_features_path):
        print(f"[warn] mllm_features_path not found: {args.mllm_features_path}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "pgap_points_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["name", "has_gt", "gt_area", "pgap_on_target", "pgap_off_target", "pgap_on_ratio"]
        if use_text_compare:
            header += ["fused_on_target", "fused_off_target", "fused_on_ratio"]
        writer.writerow(header)

        total_pgap_on = 0
        total_pgap_pts = 0
        total_fused_on = 0
        total_fused_pts = 0
        total_pos_samples = 0
        total_neg_samples = 0

        for idx in indices:
            sample = ds[int(idx)]
            img_t = sample["image"].to(device)
            gt_t = sample["mask"]
            name = sample["name"]
            gt_area = float(gt_t.sum().item())
            has_gt = int(gt_area > 0)
            if has_gt:
                total_pos_samples += 1
            else:
                total_neg_samples += 1

            text_prior = None
            compare_map = None
            compare_map_title = "Text prior + reranked points"
            fused_coords = None
            fused_labels = None
            if use_text_compare:
                try:
                    # Compute PGAP once for reranking and visualization.
                    pgap_coords, pgap_labels, pgap_sal = _run_pgap(img_t, pgap)
                    text_prior = _build_text_prior(
                        img_t,
                        sample,
                        model_vis,
                        text_conditioner_vis,
                        text_dense_prompt_vis,
                        device=device,
                        scale=args.text_dense_prompt_scale,
                    )
                    if text_prior is not None:
                        if args.text_point_mode == "refind":
                            fused_coords, fused_labels, fused_sal = _refind_points_with_text(
                                pgap,
                                pgap_sal,
                                text_prior,
                                alpha=args.text_rerank_alpha,
                                fuse_mode=args.text_saliency_fuse_mode,
                            )
                            compare_map = fused_sal.detach().cpu() if fused_sal is not None else None
                            compare_map_title = "Fused saliency + regenerated points"
                        else:
                            fused_coords, fused_labels = _rerank_points_with_text(
                                pgap_coords,
                                pgap_labels,
                                pgap_sal.detach().cpu(),
                                text_prior,
                                alpha=args.text_rerank_alpha,
                                backup_ratio=args.text_rerank_backup_ratio,
                            )
                            compare_map = text_prior
                            compare_map_title = "Text prior + reranked points"
                except Exception as e:
                    print(f"[warn] text compare failed for {name}: {e}")
                    text_prior = None
                    compare_map = None
                    fused_coords = None
                    fused_labels = None

            save_path = os.path.join(out_dir, f"{name}_pgap.png")
            metrics = vis_pgap_compare_on_sample(
                img_t,
                gt_t,
                pgap,
                save_path=save_path,
                title=name,
                show=args.show,
                compare_map=compare_map,
                compare_map_title=compare_map_title,
                fused_point_coords=fused_coords,
                fused_point_labels=fused_labels,
            )

            pgap_on = int(metrics["pgap_on"])
            pgap_total = int(metrics["pgap_total"])
            pgap_off = pgap_total - pgap_on
            pgap_on_ratio = float(pgap_on) / float(pgap_total) if pgap_total > 0 else 0.0
            row = [name, has_gt, f"{gt_area:.0f}", pgap_on, pgap_off, f"{pgap_on_ratio:.4f}"]

            total_pgap_on += pgap_on
            total_pgap_pts += pgap_total

            if metrics["fused_on"] is not None and metrics["fused_total"] is not None:
                fused_on = int(metrics["fused_on"])
                fused_total = int(metrics["fused_total"])
                fused_off = fused_total - fused_on
                fused_ratio = float(fused_on) / float(fused_total) if fused_total > 0 else 0.0
                row += [fused_on, fused_off, f"{fused_ratio:.4f}"]
                total_fused_on += fused_on
                total_fused_pts += fused_total

            writer.writerow(row)

    if total_pgap_pts > 0:
        print(f"PGAP points on GT: {total_pgap_on}/{total_pgap_pts} ({total_pgap_on / total_pgap_pts:.4f})")
    if total_fused_pts > 0:
        print(f"PGAP+Text points on GT: {total_fused_on}/{total_fused_pts} ({total_fused_on / total_fused_pts:.4f})")
    print(f"Positive samples: {total_pos_samples}, Negative samples: {total_neg_samples}")
    print(f"Saved visualizations to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
