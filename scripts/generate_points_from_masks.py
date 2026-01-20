import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def _find_with_exts(root: str, name: str, exts: List[str]) -> Optional[str]:
    base, ext = os.path.splitext(name)
    if ext.lower() in exts:
        cand1 = os.path.join(root, name)
        cand2 = os.path.join(os.path.dirname(root), name)
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
        return None
    for e in exts:
        cand1 = os.path.join(root, name + e)
        cand2 = os.path.join(os.path.dirname(root), name + e)
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
    return None


def _read_split_names(split_txt: str) -> List[str]:
    with open(split_txt, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f.readlines()]
    return [n for n in names if n and not n.startswith("#")]


def _apply_suffix(nm: str, sfx: str) -> str:
    if not sfx:
        return nm
    base, ext = os.path.splitext(nm)
    return base + sfx + ext


def _boundary_map(mask_2d: np.ndarray) -> np.ndarray:
    m = torch.from_numpy(mask_2d.astype(np.float32))[None, None, ...]
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    erode = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    b = (dil - erode).clamp(min=0.0, max=1.0)
    return (b[0, 0] > 0).cpu().numpy()


def _label_components(mask_2d: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    try:
        from scipy import ndimage as ndi
        struct = np.ones((3, 3), dtype=np.int32) if connectivity == 8 else np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.int32,
        )
        labeled, num = ndi.label(mask_2d.astype(np.uint8), structure=struct)
        return labeled, int(num)
    except Exception:
        pass
    try:
        import cv2
        num, labeled = cv2.connectedComponents(mask_2d.astype(np.uint8), connectivity=connectivity)
        return labeled, int(num - 1)
    except Exception:
        pass

    h, w = mask_2d.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 1),
                     (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    for y in range(h):
        for x in range(w):
            if mask_2d[y, x] == 0 or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask_2d[ny, nx] != 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
    return labels, label


def _centroids_from_mask(mask_2d: np.ndarray, connectivity: int = 8) -> np.ndarray:
    labels, num = _label_components(mask_2d, connectivity=connectivity)
    if num <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    sums_x = np.bincount(labs, weights=xs)
    sums_y = np.bincount(labs, weights=ys)
    counts = np.bincount(labs)
    valid = counts[1:num + 1] > 0
    cx = sums_x[1:num + 1][valid] / counts[1:num + 1][valid]
    cy = sums_y[1:num + 1][valid] / counts[1:num + 1][valid]
    return np.stack([cx, cy], axis=1).astype(np.float32)


def _sample_idx(rng: np.random.Generator, idx: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or idx.shape[0] == 0:
        return idx[:0]
    if idx.shape[0] <= n:
        return idx
    sel = rng.choice(idx.shape[0], size=n, replace=False)
    return idx[sel]


def _sample_points(
    mask_2d: np.ndarray,
    n_pos: int,
    n_neg: int,
    boundary_prior: bool,
    boundary_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.argwhere(mask_2d > 0)
    neg_idx = np.argwhere(mask_2d == 0)
    npos = min(n_pos, pos_idx.shape[0])
    nneg = min(n_neg, neg_idx.shape[0])
    if boundary_prior:
        bmap = _boundary_map(mask_2d)
        bpos = np.argwhere((mask_2d > 0) & (bmap > 0))
        bneg = np.argwhere((mask_2d == 0) & (bmap > 0))
        bp = int(npos * boundary_ratio)
        bn = int(nneg * boundary_ratio)
        pos = np.concatenate(
            [
                _sample_idx(rng, bpos, bp),
                _sample_idx(rng, pos_idx, npos - bp),
            ],
            axis=0,
        )
        neg = np.concatenate(
            [
                _sample_idx(rng, bneg, bn),
                _sample_idx(rng, neg_idx, nneg - bn),
            ],
            axis=0,
        )
    else:
        pos = _sample_idx(rng, pos_idx, npos)
        neg = _sample_idx(rng, neg_idx, nneg)
    if pos.shape[0] == 0 and neg.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    pts = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones((pos.shape[0],), dtype=np.int64), np.zeros((neg.shape[0],), dtype=np.int64)],
        axis=0,
    )
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    return xy, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split_txt", type=str, default=None, help="Optional split file; if not set, scan mask_dir.")
    p.add_argument("--mask_dir", type=str, default="masks")
    p.add_argument("--mask_suffix", type=str, default="")
    p.add_argument("--points_dir", type=str, default="points")
    p.add_argument("--mode", type=str, default="random", choices=["random", "centroid"],
                   help="random: sample pos/neg pixels; centroid: use centroid per component (+ optional neg).")
    p.add_argument("--n_pos", type=int, default=1)
    p.add_argument("--n_neg", type=int, default=4)
    p.add_argument("--boundary_prior", action="store_true")
    p.add_argument("--boundary_ratio", type=float, default=0.5)
    p.add_argument("--connectivity", type=int, default=8, choices=[4, 8],
                   help="Connected components connectivity for centroid mode.")
    p.add_argument("--points_normed", action="store_true")
    p.add_argument("--skip_bg_only", action="store_true", help="Skip masks with no foreground.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mask_root = os.path.join(args.data_root, args.mask_dir)
    points_root = os.path.join(args.data_root, args.points_dir)
    os.makedirs(points_root, exist_ok=True)

    if args.split_txt:
        split_path = args.split_txt if os.path.isabs(args.split_txt) else os.path.join(args.data_root, args.split_txt)
        names = _read_split_names(split_path)
    else:
        names = []
        for fn in os.listdir(mask_root):
            base, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXTS:
                names.append(base + ext)

    rng = np.random.default_rng(args.seed)
    written = 0
    skipped = 0
    for name in names:
        name_for_mask = name
        for pref in ("images/", "images\\"):
            if name_for_mask.startswith(pref):
                name_for_mask = name_for_mask[len(pref):]
                break
        name_with_suffix = _apply_suffix(name_for_mask, args.mask_suffix)
        mask_path = _find_with_exts(mask_root, name_with_suffix, IMG_EXTS)
        if mask_path is None:
            skipped += 1
            continue
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        mask_np = np.array(mask, dtype=np.uint8)
        mask_bin = (mask_np > 127).astype(np.uint8)
        if mask_bin.max() == 0 and args.skip_bg_only:
            skipped += 1
            continue

        if args.mode == "centroid":
            pts_pos = _centroids_from_mask(mask_bin, connectivity=args.connectivity)
            labels_pos = np.ones((pts_pos.shape[0],), dtype=np.int64)
            if args.n_neg > 0:
                neg_pts, neg_lbl = _sample_points(
                    mask_bin,
                    n_pos=0,
                    n_neg=args.n_neg,
                    boundary_prior=args.boundary_prior,
                    boundary_ratio=args.boundary_ratio,
                    rng=rng,
                )
                pts = np.concatenate([pts_pos, neg_pts], axis=0) if neg_pts.shape[0] > 0 else pts_pos
                labels = np.concatenate([labels_pos, neg_lbl], axis=0) if neg_lbl.shape[0] > 0 else labels_pos
            else:
                pts, labels = pts_pos, labels_pos
        else:
            pts, labels = _sample_points(
                mask_bin,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                boundary_prior=args.boundary_prior,
                boundary_ratio=args.boundary_ratio,
                rng=rng,
            )
        h, w = mask_bin.shape
        if args.points_normed and pts.shape[0] > 0:
            denom_w = (w - 1) if w > 1 else 1
            denom_h = (h - 1) if h > 1 else 1
            pts[:, 0] = pts[:, 0] / float(denom_w)
            pts[:, 1] = pts[:, 1] / float(denom_h)

        base = os.path.splitext(os.path.basename(mask_path))[0]
        out_path = os.path.join(points_root, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for (x, y), lab in zip(pts, labels):
                if args.points_normed:
                    f.write(f"{x:.6f} {y:.6f} {int(lab)}\n")
                else:
                    f.write(f"{int(round(x))} {int(round(y))} {int(lab)}\n")
        written += 1

    print(f"Done. Wrote {written} files, skipped {skipped}. Output: {points_root}")


if __name__ == "__main__":
    main()
