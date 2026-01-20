import os
import random
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
try:
    from torch.utils.data._utils.collate import default_collate
except Exception:
    from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as TF


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
POINT_EXTS = [".txt"]


def _find_with_exts(root: str, name: str, exts: List[str]) -> Optional[str]:
    """Find a file under a base folder, accepting:
    - plain prefix (e.g., "0001")
    - filename with extension (e.g., "0001.png")
    - path with subdirs possibly including images/ (e.g., "images/0001.png")
    Tries both under `root` and its parent directory to accommodate when `name`
    is already relative to the dataset root rather than the images/ folder.
    """
    base, ext = os.path.splitext(name)
    # Direct name with extension
    if ext.lower() in exts:
        cand1 = os.path.join(root, name)
        cand2 = os.path.join(os.path.dirname(root), name)
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
        return None

    # Name without extension: try known extensions
    for e in exts:
        cand1 = os.path.join(root, name + e)
        cand2 = os.path.join(os.path.dirname(root), name + e)
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
    return None


def _find_with_ext(root: str, name: str) -> Optional[str]:
    return _find_with_exts(root, name, IMG_EXTS)


def _load_points_txt(path: str, default_label: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    points: List[List[float]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            label = default_label
            if len(parts) >= 3:
                try:
                    label = int(float(parts[2]))
                except ValueError:
                    label = default_label
            points.append([x, y])
            labels.append(label)
    if not points:
        return (
            torch.zeros((0, 2), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class SIRSTDataset(Dataset):
    """NUDT-SIRST style dataset

    Expects:
      - split txt (train/test) with lines of image prefixes (without extension)
      - images/ and masks/ folders with matching filenames
      - optional points/ folder with per-image .txt: "x y [label]" in original pixels
    """

    def __init__(
        self,
        root: str,
        split_txt: str,
        img_dir: str = "images",
        mask_dir: str = "masks",
        size: int = 1024,
        keep_ratio_pad: bool = False,
        augment: bool = True,
        skip_bg_only: bool = False,
        mask_suffix: str = "",
        points_dir: Optional[str] = None,
        points_normed: bool = False,
        points_default_label: int = 1,
        points_required: bool = False,
        points_max: int = 0,
    ):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.size = size
        self.keep_ratio_pad = keep_ratio_pad
        self.augment = augment
        self.skip_bg_only = skip_bg_only
        self.mask_suffix = mask_suffix or ""
        self.points_dir = os.path.join(root, points_dir) if points_dir else None
        self.points_normed = bool(points_normed)
        self.points_default_label = int(points_default_label)
        self.points_required = bool(points_required)
        self.points_max = int(points_max) if points_max is not None else 0
        self.use_points = self.points_dir is not None

        txt_path = split_txt if os.path.isabs(split_txt) else os.path.join(root, split_txt)
        with open(txt_path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f.readlines()]
        self.names = [n for n in names if n and not n.startswith("#")]

        self.samples: List[Tuple[str, str, Optional[str]]] = []
        def _apply_suffix(nm: str, sfx: str) -> str:
            if not sfx:
                return nm
            base, ext = os.path.splitext(nm)
            return base + sfx + ext

        for name in self.names:
            # images: resolve as-is
            ip = _find_with_ext(self.img_dir, name)

            # masks: allow an additional suffix like "_pixels0" before extension
            # also strip a leading "images/" or "images\\" if present in txt
            name_for_mask = name
            for pref in ("images/", "images\\"):
                if name_for_mask.startswith(pref):
                    name_for_mask = name_for_mask[len(pref):]
                    break
            name_with_suffix = _apply_suffix(name_for_mask, self.mask_suffix)

            mp = _find_with_ext(self.mask_dir, name_with_suffix)
            if mp is None and self.mask_suffix:
                # fallback to original name if suffixed one not found
                mp = _find_with_ext(self.mask_dir, name_for_mask)
            if ip is None or mp is None:
                continue

            pp = None
            if self.use_points:
                name_for_point = name_for_mask
                pp = _find_with_exts(self.points_dir, name_for_point, POINT_EXTS)
                if pp is None and self.points_required:
                    continue
            self.samples.append((ip, mp, pp))
        if not self.samples:
            raise RuntimeError("No samples found. Check txt and folders.")

    def __len__(self):
        return len(self.samples)

    def _resize_square(self, img: Image.Image, mask: Image.Image):
        if not self.keep_ratio_pad:
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
            return img, mask, (0, 0, self.size, self.size)

        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), Image.BILINEAR)
        mask_r = mask.resize((nw, nh), Image.NEAREST)

        pad_l = (self.size - nw) // 2
        pad_t = (self.size - nh) // 2
        bg = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        bg_mask = Image.new("L", (self.size, self.size), 0)
        bg.paste(img_r, (pad_l, pad_t))
        bg_mask.paste(mask_r, (pad_l, pad_t))
        return bg, bg_mask, (pad_l, pad_t, nw, nh)

    def _resize_points(self, points_xy: torch.Tensor, orig_w: int, orig_h: int, roi):
        if points_xy.numel() == 0:
            return points_xy
        if not self.keep_ratio_pad:
            sx = self.size / max(1, orig_w)
            sy = self.size / max(1, orig_h)
            pts = points_xy.clone()
            pts[:, 0] = pts[:, 0] * sx
            pts[:, 1] = pts[:, 1] * sy
        else:
            pad_l, pad_t, new_w, new_h = roi
            scale = new_w / max(1, orig_w)
            pts = points_xy.clone()
            pts[:, 0] = pts[:, 0] * scale + pad_l
            pts[:, 1] = pts[:, 1] * scale + pad_t
        pts[:, 0].clamp_(0, self.size - 1)
        pts[:, 1].clamp_(0, self.size - 1)
        return pts

    def __getitem__(self, idx: int):
        img_path, mask_path, point_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        orig_w, orig_h = img.size
        points = None
        point_labels = None
        if self.use_points:
            points = torch.zeros((0, 2), dtype=torch.float32)
            point_labels = torch.zeros((0,), dtype=torch.float32)
            if point_path is not None:
                points, point_labels = _load_points_txt(point_path, self.points_default_label)
                if self.points_max > 0 and points.shape[0] > self.points_max:
                    sel = torch.randperm(points.shape[0])[: self.points_max]
                    points = points[sel]
                    point_labels = point_labels[sel]
                if self.points_normed and points.numel() > 0:
                    points = points.clone()
                    scale_w = (orig_w - 1) if orig_w > 1 else 0
                    scale_h = (orig_h - 1) if orig_h > 1 else 0
                    points[:, 0] = points[:, 0] * scale_w
                    points[:, 1] = points[:, 1] * scale_h

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
                if points is not None and points.numel() > 0:
                    points = points.clone()
                    points[:, 0] = (orig_w - 1) - points[:, 0]
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                if points is not None and points.numel() > 0:
                    points = points.clone()
                    points[:, 1] = (orig_h - 1) - points[:, 1]

        img, mask, roi = self._resize_square(img, mask)
        if points is not None:
            points = self._resize_points(points, orig_w, orig_h, roi)

        img_t = TF.to_tensor(img)  # [3,H,W], 0-1
        mask_np = np.array(mask, dtype=np.uint8)
        mask_t = torch.from_numpy((mask_np > 127).astype(np.float32))  # [H,W] 0/1

        if self.skip_bg_only and mask_t.max() == 0:
            return self.__getitem__((idx + 1) % len(self))

        sample = {
            "image": img_t,
            "mask": mask_t,
            "name": os.path.splitext(os.path.basename(img_path))[0],
            "roi": torch.tensor(roi, dtype=torch.int32),
            "orig_size": torch.tensor([self.size, self.size], dtype=torch.int32),
        }
        if self.use_points:
            sample["points"] = points
            sample["point_labels"] = point_labels
        return sample


def _pad_points_batch(points_list, labels_list):
    max_n = 0
    for p in points_list:
        if p is not None:
            max_n = max(max_n, p.shape[0])
    max_n = max(max_n, 1)
    padded_points = []
    padded_labels = []
    for p, l in zip(points_list, labels_list):
        if p is None:
            p = torch.zeros((0, 2), dtype=torch.float32)
        if l is None:
            l = torch.zeros((0,), dtype=torch.float32)
        if p.shape[0] < max_n:
            pad = max_n - p.shape[0]
            pad_pts = torch.full((pad, 2), -1.0, dtype=p.dtype)
            pad_lbl = torch.full((pad,), -1.0, dtype=l.dtype)
            p = torch.cat([p, pad_pts], dim=0)
            l = torch.cat([l, pad_lbl], dim=0)
        padded_points.append(p)
        padded_labels.append(l)
    return torch.stack(padded_points, 0), torch.stack(padded_labels, 0)


def collate_sirst(batch):
    if not batch or "points" not in batch[0]:
        return default_collate(batch)
    points_list = [b.get("points") for b in batch]
    labels_list = [b.get("point_labels") for b in batch]
    base_batch = [{k: v for k, v in b.items() if k not in ("points", "point_labels")} for b in batch]
    out = default_collate(base_batch)
    pts, lbl = _pad_points_batch(points_list, labels_list)
    out["points"] = pts
    out["point_labels"] = lbl
    return out


def make_loader(
    root,
    split_txt,
    batch_size=4,
    size=1024,
    augment=True,
    keep_ratio_pad=False,
    workers=4,
    shuffle=True,
    mask_suffix: str = "",
    points_dir: Optional[str] = None,
    points_normed: bool = False,
    points_default_label: int = 1,
    points_required: bool = False,
    points_max: int = 0,
):
    ds = SIRSTDataset(
        root,
        split_txt,
        size=size,
        augment=augment,
        keep_ratio_pad=keep_ratio_pad,
        mask_suffix=mask_suffix,
        points_dir=points_dir,
        points_normed=points_normed,
        points_default_label=points_default_label,
        points_required=points_required,
        points_max=points_max,
    )
    collate_fn = collate_sirst if ds.use_points else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=augment,
        collate_fn=collate_fn,
    )
