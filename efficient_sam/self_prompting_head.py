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
    ):
        super().__init__()
        self.top_k_pos = int(top_k_pos)
        self.top_k_neg = int(top_k_neg)
        self.min_dist = int(min_dist)

        # Lightweight segmentation head: 3-layer CNN
        # Operates at encoder feature resolution (64×64), then upsamples
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
                    # Slight negative bias on final layer so initial heatmap
                    # is close to zero (sigmoid → ~0.5), avoiding false positives
                    nn.init.constant_(m.bias, -2.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        output_size: tuple = None,
    ):
        """
        Args:
            encoder_features: [B, C, h, w] from image encoder neck
            output_size: (H, W) target resolution for heatmap, e.g. (256, 256)
        Returns:
            heatmap:      [B, 1, H, W]  sigmoid-activated probability map
            point_coords: [B, 1, K, 2]  sampled prompt point coordinates (x, y)
            point_labels: [B, 1, K]     point labels (1=pos, 0=neg)
        """
        # Predict logits at encoder resolution
        logits = self.head(encoder_features)  # [B, 1, h, w]

        # Upsample to target resolution
        if output_size is not None:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )

        heatmap = torch.sigmoid(logits)  # [B, 1, H, W]

        # Sample prompt points from heatmap
        point_coords, point_labels = self._sample_points(heatmap)

        return heatmap, point_coords, point_labels, logits

    @torch.no_grad()
    def _sample_points(self, heatmap: torch.Tensor):
        """
        Sample top-K positive and negative points from heatmap.

        Positive points: top-K highest responses with NMS-like min_dist
        Negative points: random from low-response regions

        Args:
            heatmap: [B, 1, H, W] probability map (0~1)
        Returns:
            point_coords: [B, 1, K, 2]  where K = top_k_pos + top_k_neg
            point_labels: [B, 1, K]
        """
        B, _, H, W = heatmap.shape
        device = heatmap.device
        total_k = self.top_k_pos + self.top_k_neg

        all_coords = []
        all_labels = []

        for b in range(B):
            smap = heatmap[b, 0]  # [H, W]

            # === Positive points: NMS-style top-K ===
            pos_coords = self._nms_topk(smap, self.top_k_pos)

            # === Negative points: sample from low-response regions ===
            neg_coords = self._sample_negatives(smap, self.top_k_neg)

            # Combine
            coords = torch.cat([pos_coords, neg_coords], dim=0)  # [K, 2]
            labels = torch.cat([
                torch.ones(pos_coords.shape[0], device=device, dtype=torch.long),
                torch.zeros(neg_coords.shape[0], device=device, dtype=torch.long),
            ], dim=0)  # [K]

            # Pad to fixed size if needed
            if coords.shape[0] < total_k:
                pad_len = total_k - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
                labels = F.pad(labels, (0, pad_len), value=-1)
            coords = coords[:total_k]
            labels = labels[:total_k]

            all_coords.append(coords)
            all_labels.append(labels)

        point_coords = torch.stack(all_coords, dim=0).unsqueeze(1)  # [B, 1, K, 2]
        point_labels = torch.stack(all_labels, dim=0).unsqueeze(1)  # [B, 1, K]

        return point_coords.float(), point_labels.int()

    def _nms_topk(self, smap: torch.Tensor, k: int):
        """
        Non-maximum suppression style top-K point selection.
        Ensures points are at least min_dist apart.
        """
        H, W = smap.shape
        device = smap.device

        # Use max pooling for local maximum detection
        min_d = max(1, self.min_dist)
        if min_d % 2 == 0:
            min_d += 1
        padding = min_d // 2
        local_max = F.max_pool2d(
            smap.unsqueeze(0).unsqueeze(0),
            kernel_size=min_d, stride=1, padding=padding
        ).squeeze(0).squeeze(0)

        is_peak = (smap == local_max) & (smap > 0.1)
        peak_ys, peak_xs = torch.where(is_peak)

        if peak_xs.numel() == 0:
            # Fallback: just take argmax
            flat_idx = torch.argmax(smap)
            y = flat_idx // W
            x = flat_idx % W
            return torch.stack([x, y], dim=0).unsqueeze(0).float()

        # Sort by response value and take top-K
        vals = smap[peak_ys, peak_xs]
        n_select = min(k, vals.numel())
        _, topk_idx = torch.topk(vals, n_select)
        sel_xs = peak_xs[topk_idx].float()
        sel_ys = peak_ys[topk_idx].float()

        return torch.stack([sel_xs, sel_ys], dim=1)  # [n_select, 2]

    def _sample_negatives(self, smap: torch.Tensor, k: int):
        """
        Sample negative points from low-response regions (heatmap < 0.3).
        """
        H, W = smap.shape
        device = smap.device

        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        low_mask = smap < 0.3
        low_ys, low_xs = torch.where(low_mask)

        if low_xs.numel() == 0:
            # All pixels are high response (unlikely); just pick random
            low_ys, low_xs = torch.where(smap < smap.max())
            if low_xs.numel() == 0:
                return torch.zeros((k, 2), device=device, dtype=torch.float32)

        # Random sample
        n_available = low_xs.numel()
        indices = torch.randint(0, n_available, (k,), device=device)
        sel_xs = low_xs[indices].float()
        sel_ys = low_ys[indices].float()

        return torch.stack([sel_xs, sel_ys], dim=1)  # [k, 2]


def build_self_prompting_head(
    in_channels: int = 256,
    hidden_channels: int = 64,
    top_k_pos: int = 3,
    top_k_neg: int = 2,
    min_dist: int = 8,
):
    """Factory function for SelfPromptingHead."""
    return SelfPromptingHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        top_k_pos=top_k_pos,
        top_k_neg=top_k_neg,
        min_dist=min_dist,
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

    Args:
        heatmap_logits: [B, 1, H, W] raw logits (before sigmoid)
        gt_mask: [B, 1, H, W] or [B, H, W] binary ground truth
        pos_weight: weight multiplier for positive pixels
    Returns:
        scalar loss
    """
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    # Ensure same size
    if heatmap_logits.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.float(), size=heatmap_logits.shape[-2:],
            mode="nearest"
        )

    gt_mask = gt_mask.float()

    # Weighted BCE: heavier penalty for missing targets
    weight = torch.ones_like(gt_mask)
    weight[gt_mask > 0.5] = pos_weight

    loss = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        gt_mask,
        weight=weight,
        reduction="mean",
    )
    return loss
