import argparse
import itertools
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


BLUE = (45, 115, 255)
YELLOW = (255, 196, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate binary mask images with GT-guided zoom inset and false-alarm circles."
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder containing GT.png and prediction PNGs.")
    parser.add_argument("--gt_name", type=str, default="GT.png")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=0, help="If > 0, resize all masks to target_size x target_size before annotating.")
    parser.add_argument(
        "--style_ref_size",
        type=int,
        default=0,
        help="If > 0, scale style params to match visual effect of this reference size (e.g., 256).",
    )
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--min_box", type=int, default=24)
    parser.add_argument("--inset_scale", type=float, default=2.2)
    parser.add_argument("--inset_min_w", type=int, default=110)
    parser.add_argument("--inset_max_w", type=int, default=180)
    parser.add_argument("--inset_min_h", type=int, default=60)
    parser.add_argument("--inset_max_h", type=int, default=140)
    parser.add_argument("--inset_margin", type=int, default=18)
    parser.add_argument("--inset_gap", type=int, default=24)
    parser.add_argument("--blue_width", type=int, default=5)
    parser.add_argument("--yellow_width", type=int, default=5)
    parser.add_argument("--line_width", type=int, default=5)
    parser.add_argument("--fp_circle_radius", type=int, default=28)
    parser.add_argument("--fp_merge_gap", type=int, default=8, help="Merge nearby false-alarm regions before drawing circles.")
    parser.add_argument("--fp_circle_overlap_gap", type=float, default=2.0)
    parser.add_argument("--fp_circle_abs_min_radius", type=float, default=6.0)
    parser.add_argument("--fp_circle_min_ratio", type=float, default=0.55)
    parser.add_argument("--target_circle_pad", type=float, default=2.5)
    parser.add_argument("--target_circle_min_radius", type=float, default=9.0)
    parser.add_argument("--target_circle_overlap_gap", type=float, default=2.0)
    parser.add_argument("--target_circle_abs_min_radius", type=float, default=4.0)
    parser.add_argument("--cross_target_min_ratio", type=float, default=0.80)
    parser.add_argument("--cross_fp_min_ratio", type=float, default=0.75)
    parser.add_argument("--cross_circle_gap", type=float, default=0.0)
    parser.add_argument(
        "--merge_fp_into_blue_on_overlap",
        action="store_true",
        help="If a yellow false-alarm circle overlaps a blue target circle, merge it into the blue one and hide yellow.",
    )
    return parser.parse_args()


def _scale_int(value: int, scale: float, min_v: int = 0) -> int:
    return max(min_v, int(round(float(value) * float(scale))))


def _scale_float(value: float, scale: float, min_v: float = 0.0) -> float:
    return max(min_v, float(value) * float(scale))


def apply_style_ref_scaling(args, image_shape: Tuple[int, int]):
    ref = int(getattr(args, "style_ref_size", 0) or 0)
    if ref <= 0:
        return args

    if int(getattr(args, "target_size", 0) or 0) > 0:
        cur = float(int(args.target_size))
    else:
        cur = float(max(int(image_shape[0]), int(image_shape[1])))

    scale = cur / float(ref)
    if abs(scale - 1.0) < 1e-6:
        return args

    # Circle/line visual style.
    args.blue_width = _scale_int(args.blue_width, scale, min_v=1)
    args.yellow_width = _scale_int(args.yellow_width, scale, min_v=1)
    args.line_width = _scale_int(args.line_width, scale, min_v=1)
    args.fp_circle_radius = _scale_int(args.fp_circle_radius, scale, min_v=1)
    args.target_circle_min_radius = _scale_float(args.target_circle_min_radius, scale, min_v=1.0)
    args.target_circle_abs_min_radius = _scale_float(args.target_circle_abs_min_radius, scale, min_v=1.0)
    args.fp_circle_abs_min_radius = _scale_float(args.fp_circle_abs_min_radius, scale, min_v=1.0)
    args.target_circle_pad = _scale_float(args.target_circle_pad, scale, min_v=0.0)

    # Keep inset circles visually consistent too.
    args.inset_min_w = _scale_int(args.inset_min_w, scale, min_v=8)
    args.inset_max_w = max(args.inset_min_w, _scale_int(args.inset_max_w, scale, min_v=8))
    args.inset_min_h = _scale_int(args.inset_min_h, scale, min_v=8)
    args.inset_max_h = max(args.inset_min_h, _scale_int(args.inset_max_h, scale, min_v=8))
    args.inset_margin = _scale_int(args.inset_margin, scale, min_v=0)
    args.inset_gap = _scale_int(args.inset_gap, scale, min_v=0)
    return args


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if mask.shape == target_shape:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def bbox_from_mask(mask: np.ndarray, pad: int, min_box: int) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        cx, cy = w // 2, h // 2
        half = max(min_box // 2, 12)
        return (
            max(0, cx - half),
            max(0, cy - half),
            min(w - 1, cx + half),
            min(h - 1, cy + half),
        )

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    if w < min_box:
        grow = min_box - w
        x1 -= grow // 2
        x2 += grow - grow // 2
    if h < min_box:
        grow = min_box - h
        y1 -= grow // 2
        y2 += grow - grow // 2

    max_h, max_w = mask.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(max_w - 1, x2)
    y2 = min(max_h - 1, y2)
    return int(x1), int(y1), int(x2), int(y2)


def component_rois(mask: np.ndarray, pad: int, min_box: int) -> List[Tuple[int, int, int, int]]:
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rois = []
    for label_idx in range(1, num_labels):
        comp = labels == label_idx
        if not np.any(comp):
            continue
        rois.append(bbox_from_mask(comp, pad=pad, min_box=min_box))
    if not rois:
        rois = [bbox_from_mask(mask, pad=pad, min_box=min_box)]
    rois.sort(key=lambda box: (box[0], box[1]))
    return rois


def component_rois_and_masks(
    mask: np.ndarray, pad: int, min_box: int
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], np.ndarray]]:
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    items: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], np.ndarray]] = []
    for label_idx in range(1, num_labels):
        comp = labels == label_idx
        if not np.any(comp):
            continue
        items.append(
            (
                bbox_from_mask(comp, pad=pad, min_box=min_box),
                bbox_from_mask(comp, pad=1, min_box=0),
                comp,
            )
        )
    if not items:
        items = [
            (
                bbox_from_mask(mask, pad=pad, min_box=min_box),
                bbox_from_mask(mask, pad=1, min_box=0),
                mask.copy(),
            )
        ]
    items.sort(key=lambda item: (item[0][0], item[0][1]))
    return items


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = rect
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def rect_size(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = rect
    return (x2 - x1 + 1, y2 - y1 + 1)


def rect_circle(rect: Tuple[int, int, int, int], pad: float = 0.0) -> Tuple[float, float, float]:
    cx, cy = rect_center(rect)
    w, h = rect_size(rect)
    radius = 0.5 * max(w, h) + float(pad)
    return cx, cy, radius


def false_alarm_circle(x: int, y: int, w: int, h: int, args) -> Tuple[float, float, int]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    radius = int(args.fp_circle_radius)
    return cx, cy, radius


def target_circle(rect: Tuple[int, int, int, int], args) -> Tuple[float, float, float]:
    cx, cy = rect_center(rect)
    w, h = rect_size(rect)
    radius = max(float(args.target_circle_min_radius), 0.5 * max(w, h) + float(args.target_circle_pad))
    return cx, cy, radius


def target_circle_min_radius(rect: Tuple[int, int, int, int], args) -> float:
    w, h = rect_size(rect)
    auto_min_r = 0.5 * max(w, h) + 0.8
    return max(float(args.target_circle_abs_min_radius), auto_min_r)


def false_alarm_min_radius(base_radius: float, args) -> float:
    return max(
        float(args.fp_circle_abs_min_radius),
        float(base_radius) * float(args.fp_circle_min_ratio),
    )


def shrink_overlapping_circles(
    circles: Sequence[Tuple[float, float, float]],
    min_radii: Sequence[float],
    gap: float,
    max_iter: int = 120,
) -> List[Tuple[float, float, float]]:
    work: List[List[float]] = [[float(cx), float(cy), float(r)] for cx, cy, r in circles]
    if len(work) <= 1:
        return [(cx, cy, r) for cx, cy, r in work]

    for _ in range(max_iter):
        changed = False
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                cxi, cyi, ri = work[i]
                cxj, cyj, rj = work[j]
                dist = float(np.hypot(cxi - cxj, cyi - cyj))
                required = ri + rj + gap
                if dist >= required - 1e-6:
                    continue

                overlap = required - dist
                room_i = max(0.0, ri - float(min_radii[i]))
                room_j = max(0.0, rj - float(min_radii[j]))
                total_room = room_i + room_j
                if total_room <= 1e-6:
                    continue

                shrink_i = min(room_i, overlap * (room_i / total_room))
                shrink_j = min(room_j, overlap - shrink_i)
                remain = overlap - (shrink_i + shrink_j)

                if remain > 1e-6:
                    extra_i = min(max(0.0, room_i - shrink_i), remain * 0.5)
                    shrink_i += extra_i
                    remain -= extra_i
                    extra_j = min(max(0.0, room_j - shrink_j), remain)
                    shrink_j += extra_j

                if shrink_i > 1e-6 or shrink_j > 1e-6:
                    work[i][2] = max(float(min_radii[i]), ri - shrink_i)
                    work[j][2] = max(float(min_radii[j]), rj - shrink_j)
                    changed = True
        if not changed:
            break
    return [(cx, cy, r) for cx, cy, r in work]


def resolve_target_circles(
    focus_rects: Sequence[Tuple[int, int, int, int]],
    args,
) -> List[Tuple[float, float, float]]:
    circles: List[Tuple[float, float, float]] = []
    min_radii: List[float] = []

    for rect in focus_rects:
        cx, cy, base_r = target_circle(rect, args)
        circles.append((float(cx), float(cy), float(base_r)))
        min_radii.append(float(target_circle_min_radius(rect, args)))

    return shrink_overlapping_circles(
        circles,
        min_radii=min_radii,
        gap=float(args.target_circle_overlap_gap),
    )


def resolve_false_alarm_circles(
    fp_boxes: Sequence[Tuple[int, int, int, int]],
    args,
) -> List[Tuple[float, float, float]]:
    circles: List[Tuple[float, float, float]] = []
    min_radii: List[float] = []
    for x, y, w, h in fp_boxes:
        cx, cy, radius = false_alarm_circle(x, y, w, h, args)
        base_r = float(radius)
        circles.append((float(cx), float(cy), base_r))
        min_radii.append(false_alarm_min_radius(base_r, args))
    return shrink_overlapping_circles(
        circles,
        min_radii=min_radii,
        gap=float(args.fp_circle_overlap_gap),
    )


def expand_rect(rect: Tuple[int, int, int, int], pad: int, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    w, h = image_size
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad),
    )


def rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-6:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return float(np.hypot(px - qx, py - qy))


def point_rect_distance(px: float, py: float, rect: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def point_in_rect(px: float, py: float, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    return (
        min(ax, bx) - 1e-6 <= px <= max(ax, bx) + 1e-6
        and min(ay, by) - 1e-6 <= py <= max(ay, by) + 1e-6
    )


def segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    if abs(o1) <= 1e-6 and on_segment(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= 1e-6 and on_segment(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= 1e-6 and on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= 1e-6 and on_segment(cx, cy, dx, dy, bx, by):
        return True
    return False


def segment_intersects_rect(
    seg: Tuple[float, float, float, float],
    rect: Tuple[int, int, int, int],
) -> bool:
    ax, ay, bx, by = seg
    x1, y1, x2, y2 = rect

    if point_in_rect(ax, ay, rect) or point_in_rect(bx, by, rect):
        return True

    rect_edges = [
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1),
    ]
    for edge in rect_edges:
        if segments_intersect(ax, ay, bx, by, *edge):
            return True
    return False


def false_alarm_boxes(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    merge_gap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred_mask.astype(np.uint8), connectivity=8
    )
    boxes = []
    for label_idx in range(1, num_labels):
        comp = labels == label_idx
        if np.any(comp & gt_mask):
            continue
        x, y, w, h, area = stats[label_idx]
        if area <= 0:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    if merge_gap > 0 and len(boxes) > 1:
        boxes = merge_close_boxes(boxes, gap=int(merge_gap))
    return boxes


def detected_roi_indices(pred_mask: np.ndarray, gt_components: Sequence[np.ndarray]) -> List[int]:
    detected: List[int] = []
    for idx, comp in enumerate(gt_components):
        if np.any(pred_mask & comp):
            detected.append(idx)
    return detected


def merge_boxes(boxes: Sequence[Tuple[int, int, int, int]], eps: int = 4) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        rect = (x, y, x + w - 1, y + h - 1)
        inserted = False
        for idx, cur in enumerate(merged):
            ex = expand_rect(cur, eps, (10**9, 10**9))
            if rects_overlap(rect, ex):
                merged[idx] = (
                    min(cur[0], rect[0]),
                    min(cur[1], rect[1]),
                    max(cur[2], rect[2]),
                    max(cur[3], rect[3]),
                )
                inserted = True
                break
        if not inserted:
            merged.append(rect)
    return merged


def merge_close_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    gap: int,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    gap = max(0, int(gap))
    cur = sorted((int(x), int(y), int(w), int(h)) for x, y, w, h in boxes)
    while True:
        merged_rects = merge_boxes(cur, eps=gap)
        nxt = sorted(
            (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))
            for (x1, y1, x2, y2) in merged_rects
        )
        if nxt == cur:
            return nxt
        cur = nxt


def render_mask_rgb(mask: np.ndarray) -> Image.Image:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask] = 255
    return Image.fromarray(out, mode="RGB")


def compute_inset_size(roi: Tuple[int, int, int, int], args) -> Tuple[int, int]:
    crop_w, crop_h = rect_size(roi)
    side = int(
        np.clip(
            round(max(crop_w, crop_h) * args.inset_scale),
            max(args.inset_min_w, args.inset_min_h),
            max(args.inset_max_w, args.inset_max_h),
        )
    )
    return side, side


def make_inset(image: Image.Image, roi: Tuple[int, int, int, int], args) -> Tuple[Image.Image, Tuple[int, int]]:
    x1, y1, x2, y2 = roi
    crop = image.crop((x1, y1, x2 + 1, y2 + 1))
    crop_w, crop_h = crop.size
    inset_w, inset_h = compute_inset_size(roi, args)

    inner_w = max(1, inset_w - 10)
    inner_h = max(1, inset_h - 10)
    scale = min(inner_w / max(crop_w, 1), inner_h / max(crop_h, 1))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized = crop.resize((resized_w, resized_h), resample=Image.Resampling.NEAREST)

    inset = Image.new("RGB", (inset_w, inset_h), BLACK)
    paste_x = (inset_w - resized_w) // 2
    paste_y = (inset_h - resized_h) // 2
    inset.paste(resized, (paste_x, paste_y))
    circle_mask = Image.new("L", (inset_w, inset_h), 0)
    circle_draw = ImageDraw.Draw(circle_mask)
    circle_draw.ellipse((2, 2, inset_w - 3, inset_h - 3), fill=255)
    inset = Image.composite(inset, Image.new("RGB", (inset_w, inset_h), BLACK), circle_mask)
    return inset, (inset_w, inset_h)


def corner_box(
    image_size: Tuple[int, int],
    inset_size: Tuple[int, int],
    corner: str,
    margin: int,
) -> Tuple[int, int, int, int]:
    canvas_w, canvas_h = image_size
    inset_w, inset_h = inset_size
    if corner == "tl":
        x1, y1 = margin, margin
    elif corner == "tr":
        x1, y1 = canvas_w - margin - inset_w, margin
    elif corner == "bl":
        x1, y1 = margin, canvas_h - margin - inset_h
    elif corner == "br":
        x1, y1 = canvas_w - margin - inset_w, canvas_h - margin - inset_h
    else:
        raise ValueError(f"Unsupported corner: {corner}")
    return (x1, y1, x1 + inset_w - 1, y1 + inset_h - 1)


def connector_segments(
    inset_box: Tuple[int, int, int, int],
    target_circle_value: Tuple[float, float, float],
) -> List[Tuple[float, float, float, float]]:
    icx, icy, ir = rect_circle(inset_box, pad=2.0)
    rcx, rcy, rr = target_circle_value
    dx = rcx - icx
    dy = rcy - icy
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-6:
        return []
    ux = dx / dist
    uy = dy / dist
    return [
        (
            icx + ux * ir,
            icy + uy * ir,
            rcx - ux * rr,
            rcy - uy * rr,
        )
    ]


def layout_inset_boxes(
    image_size: Tuple[int, int],
    roi_list: List[Tuple[int, int, int, int]],
    target_rects: Sequence[Tuple[int, int, int, int]],
    target_circles: Sequence[Tuple[float, float, float]],
    avoid_rects: Sequence[Tuple[int, int, int, int]],
    avoid_circles: Sequence[Tuple[float, float, float]],
    args,
) -> List[Tuple[int, int, int, int]]:
    margin = int(args.inset_margin)
    corners = ["tl", "tr", "bl", "br"]

    # Corner-only layout is the requested behavior. If there are more than four
    # targets, keep the first four left-to-right to avoid overlapping insets.
    if len(roi_list) > 4:
        roi_indices = list(range(4))
    else:
        roi_indices = list(range(len(roi_list)))

    inset_sizes = [compute_inset_size(roi_list[idx], args) for idx in roi_indices]
    best_assignment = None
    best_score = None

    for corner_perm in itertools.permutations(corners, len(roi_indices)):
        candidate_boxes = [
            corner_box(image_size, inset_sizes[pos], corner_perm[pos], margin)
            for pos in range(len(roi_indices))
        ]
        candidate_segments: List[List[Tuple[float, float, float, float]]] = [
            connector_segments(candidate_boxes[pos], target_circles[roi_indices[pos]])
            for pos in range(len(roi_indices))
        ]
        score = 0.0
        invalid = False

        for pos, roi_idx in enumerate(roi_indices):
            roi = roi_list[roi_idx]
            cand = candidate_boxes[pos]
            cand_cx, cand_cy = rect_center(cand)
            roi_cx, roi_cy = rect_center(roi)
            score += 0.08 * abs(cand_cx - roi_cx) + 0.05 * abs(cand_cy - roi_cy)

            for rect in avoid_rects:
                if rects_overlap(cand, rect):
                    score += 6e5

            for fx, fy, fr in avoid_circles:
                rect_dist = point_rect_distance(fx, fy, cand)
                if rect_dist < fr + 24:
                    score += 2.5e5 + (fr + 24 - rect_dist) * 1200.0

            for seg in candidate_segments[pos]:
                for other_idx, target_rect in enumerate(target_rects):
                    if other_idx == roi_idx:
                        continue
                    if segment_intersects_rect(seg, expand_rect(target_rect, 4, image_size)):
                        score += 1.5e6
                        invalid = True

                for other_pos, other_box in enumerate(candidate_boxes):
                    if other_pos == pos:
                        continue
                    if segment_intersects_rect(seg, expand_rect(other_box, 3, image_size)):
                        score += 7e5

                ax, ay, bx, by = seg
                for fx, fy, fr in avoid_circles:
                    dist = point_segment_distance(fx, fy, ax, ay, bx, by)
                    if dist < fr + 8:
                        score += 3e5 + (fr + 8 - dist) * 1400.0

                for other_idx, (tx, ty, tr) in enumerate(target_circles):
                    if other_idx == roi_idx:
                        continue
                    dist_to_target_circle = point_segment_distance(tx, ty, ax, ay, bx, by)
                    if dist_to_target_circle < tr + 3:
                        score += 5e5 + (tr + 3 - dist_to_target_circle) * 1600.0

        for pos_a in range(len(candidate_segments)):
            for pos_b in range(pos_a + 1, len(candidate_segments)):
                for seg_a in candidate_segments[pos_a]:
                    for seg_b in candidate_segments[pos_b]:
                        if segments_intersect(*seg_a, *seg_b):
                            score += 2.0e6
                            invalid = True

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = candidate_boxes
            if not invalid and score < 1e6:
                # Good enough corner placement; stop early.
                pass

    if best_assignment is None:
        best_assignment = [
            corner_box(image_size, inset_sizes[pos], corners[pos], margin)
            for pos in range(len(roi_indices))
        ]

    if len(roi_list) > 4:
        # Keep deterministic fallback for any remaining targets.
        extra = []
        for idx in range(4, len(roi_list)):
            extra.append(best_assignment[-1])
        return best_assignment + extra
    return best_assignment


def annotate_image(
    image: Image.Image,
    roi_list: List[Tuple[int, int, int, int]],
    focus_rects: Sequence[Tuple[int, int, int, int]],
    target_circles: Sequence[Tuple[float, float, float]],
    inset_boxes: List[Tuple[int, int, int, int]],
    fp_boxes: List[Tuple[int, int, int, int]],
    args,
    active_indices: Sequence[int] | None = None,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    if active_indices is None:
        active_indices = range(min(len(roi_list), len(inset_boxes)))

    valid_active = [
        idx for idx in active_indices
        if idx < len(roi_list) and idx < len(focus_rects) and idx < len(target_circles) and idx < len(inset_boxes)
    ]
    fp_circles = resolve_false_alarm_circles(fp_boxes, args)

    active_target_circles = [target_circles[idx] for idx in valid_active]
    active_target_pref_mins = []
    active_target_loose_mins = []
    for idx, (_, _, r) in zip(valid_active, active_target_circles):
        loose = target_circle_min_radius(focus_rects[idx], args)
        pref = max(loose, float(r) * float(args.cross_target_min_ratio))
        active_target_pref_mins.append(pref)
        active_target_loose_mins.append(loose)

    fp_pref_mins = []
    fp_loose_mins = []
    for _, _, r in fp_circles:
        loose = false_alarm_min_radius(r, args)
        pref = max(loose, float(r) * float(args.cross_fp_min_ratio))
        fp_pref_mins.append(pref)
        fp_loose_mins.append(loose)

    def has_cross_overlap(
        circles: Sequence[Tuple[float, float, float]],
        n_targets: int,
        gap: float,
    ) -> bool:
        for i in range(n_targets):
            cxi, cyi, ri = circles[i]
            for j in range(n_targets, len(circles)):
                cxj, cyj, rj = circles[j]
                if float(np.hypot(cxi - cxj, cyi - cyj)) < ri + rj + gap - 1e-6:
                    return True
        return False

    combined = active_target_circles + fp_circles
    n_targets = len(active_target_circles)
    if combined and not bool(getattr(args, "merge_fp_into_blue_on_overlap", False)):
        # Stage 1: keep circles as large as possible.
        combined = shrink_overlapping_circles(
            combined,
            active_target_pref_mins + fp_pref_mins,
            gap=float(args.cross_circle_gap),
        )
        # Stage 2: if still crossed, allow moderate extra shrink to loose minima.
        if has_cross_overlap(combined, n_targets, gap=float(args.cross_circle_gap)):
            combined = shrink_overlapping_circles(
                combined,
                active_target_loose_mins + fp_loose_mins,
                gap=float(args.cross_circle_gap),
            )

    target_circle_map = {}
    split = len(active_target_circles)
    for pos, idx in enumerate(valid_active):
        if pos < len(combined):
            target_circle_map[idx] = combined[pos]
    fp_circles = combined[split:] if len(combined) > split else []

    if bool(getattr(args, "merge_fp_into_blue_on_overlap", False)) and fp_circles:
        kept_fp_circles: List[Tuple[float, float, float]] = []
        overlap_gap = float(args.cross_circle_gap)
        for fx, fy, fr in fp_circles:
            best_idx = None
            best_dist = None
            for idx in valid_active:
                tx, ty, tr = target_circle_map.get(idx, target_circles[idx])
                dist = float(np.hypot(tx - fx, ty - fy))
                if dist <= tr + fr + overlap_gap:
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx
            if best_idx is None:
                kept_fp_circles.append((fx, fy, fr))
            else:
                tx, ty, tr = target_circle_map.get(best_idx, target_circles[best_idx])
                new_r = max(float(tr), float(np.hypot(tx - fx, ty - fy) + fr))
                target_circle_map[best_idx] = (tx, ty, new_r)
        fp_circles = kept_fp_circles

    for idx in valid_active:
        roi = roi_list[idx]
        circle_value = target_circle_map.get(idx, target_circles[idx])
        inset_box = inset_boxes[idx]
        inset_x1, inset_y1, inset_x2, inset_y2 = inset_box
        inset, _ = make_inset(image, roi, args)
        canvas.paste(inset, (inset_x1, inset_y1))

        draw.ellipse(
            (inset_x1 - 2, inset_y1 - 2, inset_x2 + 2, inset_y2 + 2),
            outline=BLUE,
            width=int(args.blue_width),
        )
        rcx, rcy, rr = circle_value
        draw.ellipse(
            (rcx - rr, rcy - rr, rcx + rr, rcy + rr),
            outline=BLUE,
            width=int(args.blue_width),
        )
        for ax, ay, bx, by in connector_segments(inset_box, circle_value):
            draw.line((ax, ay, bx, by), fill=BLUE, width=int(args.line_width))

    for cx, cy, radius in fp_circles:
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=YELLOW,
            width=int(args.yellow_width),
        )

    return canvas


def main():
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    gt_path = folder / args.gt_name
    if not gt_path.exists():
        raise FileNotFoundError(f"GT image not found: {gt_path}")

    out_dir = Path(args.out_dir) if args.out_dir else folder / "annotated_zoom_fp"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_mask = load_mask(gt_path)
    if int(args.target_size) > 0:
        gt_mask = resize_mask(gt_mask, (int(args.target_size), int(args.target_size)))
    args = apply_style_ref_scaling(args, gt_mask.shape)
    roi_items = component_rois_and_masks(gt_mask, pad=args.pad, min_box=args.min_box)
    roi_list = [roi for roi, _, _ in roi_items]
    focus_rects = [focus_rect for _, focus_rect, _ in roi_items]
    target_circles = resolve_target_circles(focus_rects, args)
    gt_components = [comp for _, _, comp in roi_items]

    image_paths = sorted(p for p in folder.glob("*.png"))
    valid_image_paths = list(image_paths)

    all_fp_rects: List[Tuple[int, int, int, int]] = []
    for image_path in valid_image_paths:
        if image_path.name == args.gt_name:
            continue
        mask = resize_mask(load_mask(image_path), gt_mask.shape)
        all_fp_rects.extend(false_alarm_boxes(mask, gt_mask, merge_gap=int(args.fp_merge_gap)))

    image_size = (gt_mask.shape[1], gt_mask.shape[0])
    avoid_rects = [expand_rect(roi, 12, image_size) for roi in roi_list]
    merged_fp_rects = merge_boxes(all_fp_rects, eps=max(6, int(args.fp_merge_gap)))
    avoid_rects.extend(expand_rect(rect, 12, image_size) for rect in merged_fp_rects)
    merged_fp_boxes = [
        (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        for (x1, y1, x2, y2) in merged_fp_rects
    ]
    avoid_circles = resolve_false_alarm_circles(merged_fp_boxes, args)

    inset_boxes = layout_inset_boxes(
        image_size=image_size,
        roi_list=roi_list,
        target_rects=roi_list,
        target_circles=target_circles,
        avoid_rects=avoid_rects,
        avoid_circles=avoid_circles,
        args=args,
    )

    for image_path in valid_image_paths:
        raw_mask = load_mask(image_path)
        if raw_mask.shape != gt_mask.shape:
            print(f"resize to GT size: {image_path} from {raw_mask.shape} to {gt_mask.shape}")
        mask = resize_mask(raw_mask, gt_mask.shape)
        image_rgb = render_mask_rgb(mask)
        fp_boxes = []
        active_indices = list(range(len(roi_list)))
        if image_path.name != args.gt_name:
            fp_boxes = false_alarm_boxes(mask, gt_mask, merge_gap=int(args.fp_merge_gap))
            active_indices = detected_roi_indices(mask, gt_components)
        annotated = annotate_image(
            image_rgb,
            roi_list,
            focus_rects,
            target_circles,
            inset_boxes,
            fp_boxes,
            args,
            active_indices=active_indices,
        )
        out_path = out_dir / image_path.name
        annotated.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
