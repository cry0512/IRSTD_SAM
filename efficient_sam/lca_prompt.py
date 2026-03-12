# -*- coding: utf-8 -*-
"""
Local Contrast Attention-guided Prompt Generator (LCA-Prompt)

核心创新点:
1. 多尺度可微分局部对比度计算, 融合红外小目标领域先验知识
2. ASG-LCA Bridge: 利用 ASG 频域滤波后的特征增强对比度计算
3. 可微分 Top-K 峰值提取, 自动生成 SAM 所需的提示点
4. Focal 辅助监督损失, 约束对比度图与 GT 对齐

论文故事:
"ASG 在频域层面执行全局的背景杂波抑制, 但频域操作是通道级的全局变换,
缺乏空间区分能力。LCA-Prompt 补充了空间维度的目标感知, 将频域滤波后
的纯净特征转化为自适应的提示信号, 形成频域滤波→空域定位→提示引导的
完整 pipeline。"
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Multi-Scale Differentiable Local Contrast Map (LCM)
# ---------------------------------------------------------------------------

class DifferentiableLCM(nn.Module):
    """多尺度可微分局部对比度图计算

    对输入图像的每个尺度, 计算:
        LCM(x,y) = |I(x,y) - mean_surround(x,y)| / (std_surround(x,y) + eps)

    使用 depthwise-conv 模拟不同尺寸环形邻域的 mean/std,
    完全可微分, 可在训练中端到端优化。

    Args:
        scales: 多尺度 LCM 卷积核大小列表, default (3, 5, 9)
        learnable_fusion: 是否使用可学习权重融合各尺度, default True
        eps: 数值稳定性, default 1e-6
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (3, 5, 9),
        learnable_fusion: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.scales = scales
        self.eps = eps
        self.learnable_fusion = learnable_fusion

        # 为每个尺度创建固定的均值卷积核 (ring-shaped mean filter)
        for k in scales:
            kernel = self._make_ring_kernel(k)
            self.register_buffer(f"kernel_{k}", kernel)

        # 可学习的尺度融合权重
        if learnable_fusion:
            self.fusion_weight = nn.Parameter(torch.ones(len(scales)) / len(scales))
        else:
            self.register_buffer(
                "fusion_weight", torch.ones(len(scales)) / len(scales)
            )

    @staticmethod
    def _make_ring_kernel(k: int) -> torch.Tensor:
        """创建环形均值核: 排除中心像素的邻域均值

        Returns:
            Tensor: shape (1, 1, k, k), 归一化的环形核
        """
        kernel = torch.ones(1, 1, k, k)
        center = k // 2
        kernel[0, 0, center, center] = 0.0  # 排除中心
        kernel = kernel / kernel.sum()
        return kernel

    def _compute_single_scale(
        self, gray: torch.Tensor, k: int
    ) -> torch.Tensor:
        """计算单个尺度的局部对比度

        Args:
            gray: [B, 1, H, W] 灰度图
            k: 卷积核大小

        Returns:
            contrast: [B, 1, H, W] 局部对比度图
        """
        pad = k // 2
        kernel = getattr(self, f"kernel_{k}").to(dtype=gray.dtype)

        # 邻域均值
        mean_surround = F.conv2d(gray, kernel, padding=pad)

        # 邻域方差 = E[X^2] - (E[X])^2
        mean_sq = F.conv2d(gray.pow(2), kernel, padding=pad)
        var_surround = (mean_sq - mean_surround.pow(2)).clamp(min=0.0)
        std_surround = torch.sqrt(var_surround + self.eps)

        # LCM = |center - mean| / std
        contrast = (gray - mean_surround).abs() / (std_surround + self.eps)
        return contrast

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """计算多尺度局部对比度图

        Args:
            images: [B, C, H, W] 输入图像 (RGB or grayscale)

        Returns:
            lcm: [B, 1, H, W] 融合后的局部对比度图, 归一化到 [0, 1]
        """
        # 转灰度
        if images.shape[1] > 1:
            gray = images.mean(dim=1, keepdim=True)
        else:
            gray = images

        # 多尺度对比度计算
        contrasts = []
        for k in self.scales:
            c = self._compute_single_scale(gray, k)
            contrasts.append(c)

        # 归一化融合权重
        if self.learnable_fusion:
            w = F.softmax(self.fusion_weight, dim=0)
        else:
            w = self.fusion_weight

        # 加权融合
        lcm = torch.zeros_like(contrasts[0])
        for i, c in enumerate(contrasts):
            lcm = lcm + w[i] * c

        # 归一化到 [0, 1]
        B = lcm.shape[0]
        flat = lcm.view(B, -1)
        minv = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        maxv = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        lcm = (lcm - minv) / (maxv - minv + self.eps)

        return lcm


# ---------------------------------------------------------------------------
# 2.  ASG-Enhanced Contrast Bridge
# ---------------------------------------------------------------------------

class ASGContrastBridge(nn.Module):
    """ASG 增强对比度桥接模块

    将 encoder neck 输出 (已经过 ASG 频域滤波) 投影为单通道 "纯净度图",
    与原始图像的 LCM 融合, 使对比度计算受益于 ASG 的杂波抑制。

    融合方式: enhanced_lcm = lcm * (1 + alpha * purity_map)

    Args:
        neck_dim: encoder neck 特征通道数, default 256
        init_alpha: 初始融合强度, default 0.5
    """

    def __init__(self, neck_dim: int = 256, init_alpha: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_dim // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(
        self,
        lcm: torch.Tensor,
        neck_features: torch.Tensor,
    ) -> torch.Tensor:
        """将 ASG neck 特征与 LCM 融合

        Args:
            lcm: [B, 1, H, W] 局部对比度图
            neck_features: [B, C, h, w] encoder neck 输出 (h,w 通常远小于 H,W)

        Returns:
            enhanced_lcm: [B, 1, H, W] ASG 增强后的对比度图
        """
        purity = self.proj(neck_features)  # [B, 1, h, w]

        # 上采样到 LCM 分辨率
        if tuple(purity.shape[-2:]) != tuple(lcm.shape[-2:]):
            purity = F.interpolate(
                purity,
                size=tuple(lcm.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )

        # 乘性融合: LCM * (1 + alpha * purity)
        alpha = torch.sigmoid(self.alpha)  # 约束在 (0, 1)
        enhanced = lcm * (1.0 + alpha * purity)

        # 重归一化
        B = enhanced.shape[0]
        flat = enhanced.view(B, -1)
        minv = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        maxv = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        enhanced = (enhanced - minv) / (maxv - minv + 1e-6)

        return enhanced


# ---------------------------------------------------------------------------
# 3.  Soft Top-K Point Extractor
# ---------------------------------------------------------------------------

class SoftTopKExtractor(nn.Module):
    """自适应 Top-K 峰值提取

    从对比度图中提取响应位置作为 SAM 提示点。
    使用自适应阈值动态决定每张图的有效点数:
        - 先做 NMS 找到所有局部极大值
        - 取 Top-K 候选
        - 用自适应阈值 (max_val * adaptive_ratio) 过滤低置信度的点
        - 每张图保留 [1, top_k] 个有效点, 其余填充为 -1

    这样单目标图像通常只产生 1~2 个点, 多目标图像才会产生更多点。

    Args:
        top_k: 最大提示点数, default 5
        min_dist: 局部极大值最小距离, default 8
        dynamic_thr_quantile: NMS 入口阈值分位数, default 0.9
        adaptive_ratio: 自适应截断比例 (相对于最高峰值), default 0.5
            例如 0.5 表示只保留对比度 >= 最高峰值 50% 的点
    """

    def __init__(
        self,
        top_k: int = 5,
        min_dist: int = 8,
        dynamic_thr_quantile: float = 0.9,
        adaptive_ratio: float = 0.5,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.min_dist = int(min_dist)
        self.dynamic_thr_quantile = float(dynamic_thr_quantile)
        self.adaptive_ratio = float(adaptive_ratio)

    def _extract_peaks(self, contrast_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从对比度图中提取峰值点 (自适应点数)

        Args:
            contrast_map: [B, 1, H, W] 对比度图

        Returns:
            point_coords: [B, K, 2] 提示点坐标 (x, y), 无效点填充为 -1
            point_labels: [B, K] 提示点标签, 有效=1, 无效=-1
        """
        B, _, H, W = contrast_map.shape
        device = contrast_map.device

        # NMS: 局部极大值
        k_size = max(3, self.min_dist)
        if k_size % 2 == 0:
            k_size += 1
        padding = k_size // 2
        local_max = F.max_pool2d(contrast_map, kernel_size=k_size, stride=1, padding=padding)

        # 动态阈值 (NMS 入口门槛)
        flat = contrast_map.view(B, -1)
        n = flat.shape[1]
        k_thr = max(1, int((1.0 - self.dynamic_thr_quantile) * n))
        vals_topk, _ = torch.topk(flat, k_thr, dim=1)
        dyn_thr = vals_topk[:, -1].view(B, 1, 1, 1)

        is_peak = (contrast_map == local_max) & (contrast_map > dyn_thr)

        coords_list = []
        labels_list = []
        for b in range(B):
            peak_map = is_peak[b, 0]  # [H, W]
            y_idx, x_idx = torch.where(peak_map)

            if x_idx.numel() == 0:
                # Fallback: 取全局最大值 (保证至少 1 个点)
                flat_idx = torch.argmax(contrast_map[b, 0])
                fy = flat_idx // W
                fx = flat_idx % W
                points = torch.stack([fx, fy], dim=0).unsqueeze(0).float()
                n_valid = 1
            else:
                # 按对比度值排序取 top-K 候选
                vals = contrast_map[b, 0, y_idx, x_idx]
                k = min(self.top_k, vals.numel())
                topk_vals, topk_indices = torch.topk(vals, k)
                points = torch.stack(
                    [x_idx[topk_indices], y_idx[topk_indices]], dim=1
                ).float()

                # ---- 自适应截断 ----
                # 只保留对比度 >= 最高峰值 * adaptive_ratio 的点
                peak_max = topk_vals[0]  # 最高对比度
                adaptive_thr = peak_max * self.adaptive_ratio
                valid_mask = topk_vals >= adaptive_thr
                n_valid = max(1, int(valid_mask.sum().item()))  # 至少保留 1 个
                points = points[:n_valid]

            # 填充到 top_k, 无效位置标为 -1
            if points.shape[0] < self.top_k:
                pad_len = self.top_k - points.shape[0]
                points = F.pad(points, (0, 0, 0, pad_len), value=-1.0)
            else:
                points = points[: self.top_k]
                n_valid = min(n_valid, self.top_k)

            labels = torch.full((self.top_k,), -1.0, device=device)
            labels[:n_valid] = 1.0  # 仅有效点为正

            coords_list.append(points)
            labels_list.append(labels)

        point_coords = torch.stack(coords_list, 0)  # [B, K, 2]
        point_labels = torch.stack(labels_list, 0)    # [B, K]
        return point_coords, point_labels

    def label_by_gt(
        self,
        point_coords: torch.Tensor,
        gt_mask: torch.Tensor,
        neg_from_contrast: bool = True,
        contrast_map: Optional[torch.Tensor] = None,
        max_neg: int = 2,
    ) -> torch.Tensor:
        """根据 GT mask 为提取的点分配正/负标签

        点在 GT 内 → pos (1)
        点在 GT 外 → neg (0)
        填充点保持 -1

        Args:
            point_coords: [B, K, 2] (x, y) format
            gt_mask: [B, H, W] GT mask
            neg_from_contrast: 是否从高对比度非目标区域额外采样负样本
            contrast_map: [B, 1, H, W] 对比度图 (neg_from_contrast=True 时需要)
            max_neg: 最大负样本数

        Returns:
            point_labels: [B, K] 重新标注后的标签
        """
        B, K, _ = point_coords.shape
        device = point_coords.device
        _, H, W = gt_mask.shape

        labels = torch.full((B, K), -1.0, device=device)
        for b in range(B):
            for i in range(K):
                x, y = point_coords[b, i]
                if x < 0 or y < 0:
                    continue  # 填充点保持 -1
                xi = int(x.round().clamp(0, W - 1).item())
                yi = int(y.round().clamp(0, H - 1).item())
                labels[b, i] = 1.0 if gt_mask[b, yi, xi] > 0 else 0.0

        return labels

    def forward(
        self,
        contrast_map: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取提示点

        Args:
            contrast_map: [B, 1, H, W] 对比度图
            gt_mask: [B, H, W] 可选 GT mask, 提供时用于标注正负样本

        Returns:
            point_coords: [B, K, 2] (x, y) format
            point_labels: [B, K] labels (1=pos, 0=neg, -1=padding)
        """
        point_coords, point_labels = self._extract_peaks(contrast_map)

        if gt_mask is not None:
            point_labels = self.label_by_gt(
                point_coords, gt_mask,
                contrast_map=contrast_map,
            )

        return point_coords, point_labels


# ---------------------------------------------------------------------------
# 4.  LCA Supervision Loss
# ---------------------------------------------------------------------------

class LCASupervisionLoss(nn.Module):
    """LCA 对比度图的辅助监督损失

    约束 LCA 对比度图应在 GT 目标区域有高响应, 背景区域低响应。
    使用 Focal Loss 形式处理极端的正负样本不平衡。

    Args:
        alpha: Focal Loss 的 alpha 参数 (正样本权重), default 0.75
        gamma: Focal Loss 的聚焦参数, default 2.0
        dice_weight: Dice Loss 分量权重, default 0.5
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)

    def focal_bce(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Focal Binary Cross-Entropy Loss

        Args:
            pred: [B, 1, H, W] 预测值 (0~1)
            target: [B, 1, H, W] 目标值 {0, 1}
        """
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = -(
            self.alpha * target * torch.log(pred)
            + (1 - self.alpha) * (1 - target) * torch.log(1 - pred)
        )
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()

    def dice(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Dice Loss"""
        inter = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()

    def forward(
        self, contrast_map: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算 LCA 辅助损失

        Args:
            contrast_map: [B, 1, H, W] 对比度图 (已归一化到 [0,1])
            gt_mask: [B, H, W] or [B, 1, H, W] GT mask

        Returns:
            loss: 标量损失
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        gt_mask = gt_mask.float()

        # 确保尺寸匹配
        if tuple(contrast_map.shape[-2:]) != tuple(gt_mask.shape[-2:]):
            contrast_map = F.interpolate(
                contrast_map,
                size=tuple(gt_mask.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )

        focal = self.focal_bce(contrast_map, gt_mask)
        dice = self.dice(contrast_map, gt_mask)
        return focal + self.dice_weight * dice


# ---------------------------------------------------------------------------
# 5.  LCA Prompt Generator (Main Entry)
# ---------------------------------------------------------------------------

class LCAPromptGenerator(nn.Module):
    """局部对比度引导的自适应提示生成器

    将 DifferentiableLCM, ASGContrastBridge, SoftTopKExtractor 组合为
    统一的提示生成模块。

    Pipeline:
        输入图像 → 多尺度 LCM → (可选) ASG Bridge 增强 → Top-K 峰值提取 → 提示点

    Args:
        scales: LCM 卷积核尺寸, default (3, 5, 9)
        top_k: 提取的提示点数, default 5
        min_dist: 峰值最小距离, default 8
        use_asg_bridge: 是否启用 ASG 增强, default True
        neck_dim: encoder neck 通道数, default 256
        sup_alpha: Focal Loss alpha, default 0.75
        sup_gamma: Focal Loss gamma, default 2.0
        sup_dice_weight: Dice 分量权重, default 0.5
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (3, 5, 9),
        top_k: int = 5,
        min_dist: int = 8,
        adaptive_ratio: float = 0.5,
        use_asg_bridge: bool = True,
        neck_dim: int = 256,
        sup_alpha: float = 0.75,
        sup_gamma: float = 2.0,
        sup_dice_weight: float = 0.5,
    ):
        super().__init__()
        self.lcm = DifferentiableLCM(scales=scales, learnable_fusion=True)
        self.extractor = SoftTopKExtractor(
            top_k=top_k, min_dist=min_dist, adaptive_ratio=adaptive_ratio
        )
        self.use_asg_bridge = bool(use_asg_bridge)
        if self.use_asg_bridge:
            self.asg_bridge = ASGContrastBridge(neck_dim=neck_dim)
        else:
            self.asg_bridge = None
        self.sup_loss = LCASupervisionLoss(
            alpha=sup_alpha, gamma=sup_gamma, dice_weight=sup_dice_weight
        )

    def forward(
        self,
        images: torch.Tensor,
        neck_features: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """生成 LCA 提示点

        Args:
            images: [B, C, H, W] 输入图像
            neck_features: [B, C_neck, h, w] 可选 encoder neck 输出 (ASG 滤波后)
            gt_mask: [B, H, W] 可选 GT mask (训练时提供)

        Returns:
            point_coords: [B, K, 2] 提示点坐标 (x, y)
            point_labels: [B, K] 提示点标签
            contrast_map: [B, 1, H, W] 对比度图
            lca_loss: 标量辅助损失 (仅在 gt_mask 提供时返回, 否则 None)
        """
        # Step 1: 多尺度 LCM
        contrast_map = self.lcm(images)

        # Step 2: ASG Bridge 增强 (可选)
        if self.use_asg_bridge and self.asg_bridge is not None and neck_features is not None:
            contrast_map = self.asg_bridge(contrast_map, neck_features)

        # Step 3: Top-K 峰值提取
        point_coords, point_labels = self.extractor(contrast_map, gt_mask=gt_mask)

        # Step 4: 辅助监督损失
        lca_loss = None
        if gt_mask is not None:
            lca_loss = self.sup_loss(contrast_map, gt_mask)

        return point_coords, point_labels, contrast_map, lca_loss

    def extra_repr(self) -> str:
        return (
            f"scales={self.lcm.scales}, "
            f"top_k={self.extractor.top_k}, "
            f"min_dist={self.extractor.min_dist}, "
            f"asg_bridge={self.use_asg_bridge}"
        )
