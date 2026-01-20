import os
import sys
import argparse
import csv

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sirst_dataset import SIRSTDataset
from efficient_sam.PGAP import PhasePromptGenerator


@torch.no_grad()
def vis_pgap_on_sample(img_t, gt_t, pgap: PhasePromptGenerator, save_path=None, title="", show=False):
    """
    img_t: torch.Tensor, shape [C,H,W]
    gt_t : torch.Tensor or np.ndarray, shape [H,W], 0/1
    """
    if isinstance(gt_t, torch.Tensor):
        gt = gt_t.detach().cpu().numpy()
    else:
        gt = gt_t
    gt = (gt > 0).astype(np.uint8)

    point_coords, point_labels, saliency = pgap(img_t.unsqueeze(0))  # [1,K,2], [1,K], [1,1,H,W]
    pts = point_coords[0].detach().cpu().numpy()  # Kx2 (x,y)
    lbl = point_labels[0].detach().cpu().numpy()
    sal = saliency[0, 0].detach().cpu().numpy()   # HxW

    img = img_t.detach().cpu().numpy()
    if img.ndim == 3:
        base = img[0]
    else:
        base = img
    base = base - base.min()
    base = base / (base.max() + 1e-6)

    H, W = base.shape[-2], base.shape[-1]
    valid = lbl >= 0
    pts = pts[valid]
    if pts.size == 0:
        xs = np.array([], dtype=int)
        ys = np.array([], dtype=int)
    else:
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    on_target = gt[ys, xs] > 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(base, cmap="gray")
    axes[0].scatter(xs[on_target], ys[on_target], s=35, marker="o", c="lime", label="pts on GT")
    axes[0].scatter(xs[~on_target], ys[~on_target], s=35, marker="x", c="red", label="pts on BG")
    axes[0].set_title("Image + PGAP points")
    axes[0].legend(loc="lower right")
    axes[0].axis("off")

    axes[1].imshow(sal, cmap="hot")
    axes[1].scatter(xs[on_target], ys[on_target], s=35, marker="o", c="lime")
    axes[1].scatter(xs[~on_target], ys[~on_target], s=35, marker="x", c="red")
    axes[1].set_title("PGAP saliency + points")
    axes[1].axis("off")

    axes[2].imshow(base, cmap="gray")
    axes[2].contour(gt.astype(float), levels=[0.5], linewidths=1)
    axes[2].set_title("GT contour")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return int(on_target.sum()), int(len(on_target))


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

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "pgap_points_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "has_gt", "gt_area", "on_target", "off_target", "on_ratio"])

        total_on = 0
        total_pts = 0
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

            save_path = os.path.join(out_dir, f"{name}_pgap.png")
            on_cnt, total_cnt = vis_pgap_on_sample(
                img_t, gt_t, pgap, save_path=save_path, title=name, show=args.show
            )
            off_cnt = total_cnt - on_cnt
            on_ratio = float(on_cnt) / float(total_cnt) if total_cnt > 0 else 0.0
            writer.writerow([name, has_gt, f"{gt_area:.0f}", on_cnt, off_cnt, f"{on_ratio:.4f}"])
            total_on += on_cnt
            total_pts += total_cnt

    if total_pts > 0:
        print(f"Total points on GT: {total_on}/{total_pts} ({total_on / total_pts:.4f})")
    print(f"Positive samples: {total_pos_samples}, Negative samples: {total_neg_samples}")
    print(f"Saved visualizations to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
