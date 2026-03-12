"""
Self-Prompting Head for EfficientSAM IRSTD.

Replaces manual/GT point prompts with a learnable detection head
that predicts target heatmaps from encoder features, enabling
fully automatic end-to-end infrared small target detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfPromptingHead(nn.Module):
    """
    Lightweight detection head that predicts target heatmaps from
    the image encoder's feature map, then samples prompt points.

    Input:  encoder features  [B, C, h, w]  (typically [B, 256, 64, 64])
    Output: heatmap           [B, 1, H, W]  (original image resolution)
            point_coords      [B, 1, K, 2]  (x, y in pixel coords)
            point_labels      [B, 1, K]     (1=positive, 0=negative)
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        top_k_pos: int = 3,
        top_k_neg: int = 2,
        min_dist: int = 8,
        peak_thr: float = 0.1,
        low_response_thr: float = 0.3,
    ):
        super().__init__()
        self.top_k_pos = int(top_k_pos)
        self.top_k_neg = int(top_k_neg)
        self.min_dist = int(min_dist)
        self.peak_thr = float(peak_thr)
        self.low_response_thr = float(low_response_thr)

        # Lightweight segmentation head: 3-layer CNN.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Small initialization to avoid disrupting pretrained encoder."""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Start with a conservative heatmap to limit early false alarms.
                    nn.init.constant_(m.bias, -2.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        output_size: tuple = None,
        gt_mask: torch.Tensor = None,
    ):
        """
        Args:
            encoder_features: [B, C, h, w] from image encoder neck
            output_size: (H, W) target resolution for heatmap, e.g. (256, 256)
            gt_mask: optional GT mask used during training for hard-negative mining
        Returns:
            heatmap:      [B, 1, H, W] sigmoid-activated probability map
            point_coords: [B, 1, K, 2] sampled prompt point coordinates (x, y)
            point_labels: [B, 1, K] point labels (1=pos, 0=neg)
        """
        logits = self.head(encoder_features)

        if output_size is not None:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )

        heatmap = torch.sigmoid(logits)
        point_coords, point_labels = self._sample_points(heatmap, gt_mask=gt_mask)
        return heatmap, point_coords, point_labels, logits

    @torch.no_grad()
    def _sample_points(self, heatmap: torch.Tensor, gt_mask: torch.Tensor = None):
        """
        Sample top-K positive and negative points from heatmap.

        Positive points: top-K peaks with NMS-like spacing.
        Negative points:
          - Training: high-response false-alarm peaks outside GT first.
          - Fallback / inference: low-response regions.
        """
        bsz, _, h, w = heatmap.shape
        device = heatmap.device
        total_k = self.top_k_pos + self.top_k_neg

        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            if gt_mask.shape[-2:] != (h, w):
                gt_mask = F.interpolate(gt_mask.float(), size=(h, w), mode="nearest")
            gt_mask = gt_mask.float()

        all_coords = []
        all_labels = []
        for b in range(bsz):
            smap = heatmap[b, 0]
            gt_mask_b = gt_mask[b, 0] if gt_mask is not None else None

            peak_coords, _ = self._extract_peaks(
                smap,
                max_k=max(self.top_k_pos + self.top_k_neg * 4, self.top_k_pos),
            )
            pos_coords = peak_coords[: min(self.top_k_pos, peak_coords.shape[0])]
            neg_coords = self._sample_negatives(
                smap,
                self.top_k_neg,
                peak_coords=peak_coords,
                gt_mask=gt_mask_b,
                exclude_coords=pos_coords,
            )

            coords = torch.cat([pos_coords, neg_coords], dim=0)
            labels = torch.cat(
                [
                    torch.ones(pos_coords.shape[0], device=device, dtype=torch.long),
                    torch.zeros(neg_coords.shape[0], device=device, dtype=torch.long),
                ],
                dim=0,
            )

            if coords.shape[0] < total_k:
                pad_len = total_k - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
                labels = F.pad(labels, (0, pad_len), value=-1)
            coords = coords[:total_k]
            labels = labels[:total_k]

            all_coords.append(coords)
            all_labels.append(labels)

        point_coords = torch.stack(all_coords, dim=0).unsqueeze(1)
        point_labels = torch.stack(all_labels, dim=0).unsqueeze(1)
        return point_coords.float(), point_labels.int()

    def _extract_peaks(self, smap: torch.Tensor, max_k: int = None):
        """Extract sorted local maxima with NMS-like spacing."""
        _, w = smap.shape

        min_d = max(1, self.min_dist)
        if min_d % 2 == 0:
            min_d += 1
        padding = min_d // 2
        local_max = F.max_pool2d(
            smap.unsqueeze(0).unsqueeze(0),
            kernel_size=min_d,
            stride=1,
            padding=padding,
        ).squeeze(0).squeeze(0)

        is_peak = (smap == local_max) & (smap > self.peak_thr)
        peak_ys, peak_xs = torch.where(is_peak)
        if peak_xs.numel() == 0:
            flat_idx = torch.argmax(smap)
            y = flat_idx // w
            x = flat_idx % w
            coords = torch.stack([x, y], dim=0).unsqueeze(0).float()
            scores = smap[y, x].unsqueeze(0)
            return coords, scores

        vals = smap[peak_ys, peak_xs]
        n_select = vals.numel() if max_k is None else min(max_k, vals.numel())
        _, topk_idx = torch.topk(vals, n_select)
        coords = torch.stack([peak_xs[topk_idx].float(), peak_ys[topk_idx].float()], dim=1)
        scores = vals[topk_idx]
        return coords, scores

    def _sample_negatives(
        self,
        smap: torch.Tensor,
        k: int,
        peak_coords: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        exclude_coords: torch.Tensor = None,
    ):
        """Prefer hard negatives from high-response false alarms outside GT."""
        device = smap.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        if peak_coords is None:
            peak_coords, _ = self._extract_peaks(
                smap,
                max_k=max(k * 4, self.top_k_pos + k),
            )

        selected = torch.zeros((0, 2), device=device, dtype=torch.float32)
        if gt_mask is not None and peak_coords.numel() > 0:
            peak_x = peak_coords[:, 0].long().clamp(min=0, max=smap.shape[1] - 1)
            peak_y = peak_coords[:, 1].long().clamp(min=0, max=smap.shape[0] - 1)
            outside_gt = gt_mask[peak_y, peak_x] <= 0.5
            if exclude_coords is not None and exclude_coords.numel() > 0:
                same_x = peak_coords[:, None, 0] == exclude_coords[None, :, 0]
                same_y = peak_coords[:, None, 1] == exclude_coords[None, :, 1]
                outside_gt = outside_gt & (~(same_x & same_y).any(dim=1))
            hard_neg = peak_coords[outside_gt]
            if hard_neg.shape[0] > 0:
                selected = hard_neg[: min(k, hard_neg.shape[0])]

        if selected.shape[0] >= k:
            return selected[:k]

        if exclude_coords is not None and exclude_coords.numel() > 0:
            exclude_all = torch.cat([exclude_coords, selected], dim=0)
        else:
            exclude_all = selected
        fallback = self._sample_low_response_negatives(
            smap,
            k - selected.shape[0],
            exclude_coords=exclude_all,
        )
        if selected.numel() == 0:
            return fallback
        return torch.cat([selected, fallback], dim=0)

    def _sample_low_response_negatives(
        self,
        smap: torch.Tensor,
        k: int,
        exclude_coords: torch.Tensor = None,
    ):
        device = smap.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        low_mask = smap < self.low_response_thr
        if exclude_coords is not None and exclude_coords.numel() > 0:
            ex = exclude_coords.long()
            ex_x = ex[:, 0].clamp(min=0, max=smap.shape[1] - 1)
            ex_y = ex[:, 1].clamp(min=0, max=smap.shape[0] - 1)
            low_mask[ex_y, ex_x] = False
        low_ys, low_xs = torch.where(low_mask)

        if low_xs.numel() == 0:
            low_ys, low_xs = torch.where(smap < smap.max())
            if exclude_coords is not None and exclude_coords.numel() > 0 and low_xs.numel() > 0:
                keep = torch.ones_like(low_xs, dtype=torch.bool)
                ex = exclude_coords.long()
                for idx in range(ex.shape[0]):
                    keep = keep & ~((low_xs == ex[idx, 0]) & (low_ys == ex[idx, 1]))
                low_xs = low_xs[keep]
                low_ys = low_ys[keep]
            if low_xs.numel() == 0:
                return torch.zeros((0, 2), device=device, dtype=torch.float32)

        n_select = min(k, low_xs.numel())
        indices = torch.randperm(low_xs.numel(), device=device)[:n_select]
        return torch.stack([low_xs[indices].float(), low_ys[indices].float()], dim=1)


def build_self_prompting_head(
    in_channels: int = 256,
    hidden_channels: int = 64,
    top_k_pos: int = 3,
    top_k_neg: int = 2,
    min_dist: int = 8,
    peak_thr: float = 0.1,
    low_response_thr: float = 0.3,
):
    """Factory function for SelfPromptingHead."""
    return SelfPromptingHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        top_k_pos=top_k_pos,
        top_k_neg=top_k_neg,
        min_dist=min_dist,
        peak_thr=peak_thr,
        low_response_thr=low_response_thr,
    )


def self_prompt_heatmap_loss(
    heatmap_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 10.0,
):
    """
    Weighted BCE loss for heatmap supervision (AMP-safe).

    Uses binary_cross_entropy_with_logits which is safe under autocast.
    Because targets are tiny (< 1% of pixels), positive pixels
    are weighted much higher than negatives.
    """
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    if heatmap_logits.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.float(), size=heatmap_logits.shape[-2:], mode="nearest"
        )

    gt_mask = gt_mask.float()
    weight = torch.ones_like(gt_mask)
    weight[gt_mask > 0.5] = pos_weight

    loss = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        gt_mask,
        weight=weight,
        reduction="mean",
    )
    return loss
