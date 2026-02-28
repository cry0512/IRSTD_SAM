# -*- coding: utf-8 -*-
"""
Frequency-Aware Boundary Loss (FAB Loss) for Infrared Small Target Detection

核心创新点:
1. 仅在边界区域计算频率损失，聚焦小目标边缘
2. 高频分量加权，强化边界细节监督
3. 多尺度边界提取，覆盖不同厚度的边界

论文故事:
"传统频率一致性损失对整个mask计算，小目标边界信息被大面积背景稀释。
我们提出频率感知边界损失，仅在边界区域进行频域分析，
并对高频分量加权，显式强化小目标边缘的学习。"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAwareBoundaryLoss(nn.Module):
    """
    频率感知边界损失
    
    Args:
        num_bins: 径向频率分箱数量
        boundary_width: 边界宽度 (膨胀-腐蚀的 kernel size)
        high_freq_weight: 高频分量权重因子
        use_multiscale: 是否使用多尺度边界
        eps: 数值稳定性
    """
    
    def __init__(
        self,
        num_bins: int = 16,
        boundary_width: int = 3,
        high_freq_weight: float = 2.0,
        use_multiscale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.boundary_width = boundary_width
        self.high_freq_weight = high_freq_weight
        self.use_multiscale = use_multiscale
        self.eps = eps
        
        # 预计算高频权重向量
        # 低频权重=1，高频权重逐渐增加到 high_freq_weight
        freq_weights = torch.linspace(1.0, high_freq_weight, num_bins)
        self.register_buffer('freq_weights', freq_weights)
    
    def extract_boundary(self, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        提取边界区域: dilate(mask) - erode(mask)
        
        Args:
            mask: [B, 1, H, W] 二值 mask
            kernel_size: 膨胀/腐蚀核大小
            
        Returns:
            boundary: [B, 1, H, W] 边界区域
        """
        padding = kernel_size // 2
        
        # 膨胀
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        
        # 腐蚀 (对 1-mask 膨胀再取反)
        eroded = 1 - F.max_pool2d(1 - mask, kernel_size, stride=1, padding=padding)
        
        # 边界 = 膨胀 - 腐蚀
        boundary = dilated - eroded
        
        return boundary.clamp(0, 1)
    
    def extract_multiscale_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        多尺度边界提取: 融合不同宽度的边界
        
        Returns:
            boundary: [B, 1, H, W] 多尺度边界
        """
        boundaries = []
        for k in [3, 5, 7]:
            b = self.extract_boundary(mask, kernel_size=k)
            boundaries.append(b)
        
        # 取并集
        boundary = torch.stack(boundaries, dim=0).max(dim=0)[0]
        return boundary
    
    def compute_boundary_frequency_profile(
        self,
        mask: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算边界区域的径向频率能量谱
        
        Args:
            mask: [B, 1, H, W] 预测或GT mask
            boundary: [B, 1, H, W] 边界区域
            
        Returns:
            profile: [B, 1, num_bins] 径向频率能量谱
        """
        # 仅保留边界区域
        boundary_mask = mask * boundary
        
        B, C, H, W = boundary_mask.shape
        
        # FFT
        spec = torch.fft.rfft2(boundary_mask.float(), dim=(-2, -1), norm="forward")
        energy = spec.real.pow(2) + spec.imag.pow(2)
        
        # 计算径向频率
        fy = torch.fft.fftfreq(H, d=1.0, device=mask.device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=mask.device)
        fy = fy.to(mask.dtype).view(1, 1, H, 1)
        fx = fx.to(mask.dtype).view(1, 1, 1, fx.numel())
        
        radius = torch.sqrt(fy.pow(2) + fx.pow(2))
        radius = radius / (radius.max().clamp(min=self.eps))
        
        # 分箱
        bin_idx = torch.clamp((radius * (self.num_bins - 1)).long(), max=self.num_bins - 1)
        
        # 统计每个频率 bin 的能量
        energy_flat = energy.reshape(B, C, -1)
        idx_flat = bin_idx.reshape(1, 1, -1).expand_as(energy_flat)
        
        profile = torch.zeros(B, C, self.num_bins, device=mask.device, dtype=energy.dtype)
        profile.scatter_add_(2, idx_flat, energy_flat)
        
        # 归一化
        counts = torch.zeros(self.num_bins, device=mask.device, dtype=energy.dtype)
        counts.scatter_add_(0, bin_idx.reshape(-1), 
                           torch.ones(bin_idx.numel(), device=mask.device, dtype=energy.dtype))
        counts = counts.clamp_min_(1.0)
        
        profile = profile / counts.view(1, 1, -1)
        profile = profile / (profile.sum(dim=-1, keepdim=True) + self.eps)
        
        return profile
    
    def forward(
        self,
        pred: torch.Tensor,  # [B, 1, H, W] 预测 logits 或 概率
        gt: torch.Tensor,    # [B, 1, H, W] GT mask
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        """
        计算频率感知边界损失
        
        Args:
            pred: 预测 mask (logits 或概率)
            gt: GT mask
            apply_sigmoid: 是否对 pred 应用 sigmoid
            
        Returns:
            loss: 标量损失
        """
        if apply_sigmoid:
            pred = torch.sigmoid(pred)
        
        gt = gt.float()
        
        # 提取边界
        if self.use_multiscale:
            gt_boundary = self.extract_multiscale_boundary(gt)
        else:
            gt_boundary = self.extract_boundary(gt, self.boundary_width)
        
        # 计算边界区域的频率谱
        pred_profile = self.compute_boundary_frequency_profile(pred, gt_boundary)
        gt_profile = self.compute_boundary_frequency_profile(gt, gt_boundary)
        
        # 高频加权 L1 损失
        freq_weights = self.freq_weights.view(1, 1, -1).to(pred.device)
        weighted_diff = torch.abs(pred_profile - gt_profile) * freq_weights
        
        loss = weighted_diff.mean()
        
        return loss


class HybridBoundaryLoss(nn.Module):
    """
    混合边界损失: 频率感知边界损失 + 空域边界损失
    
    同时在空域和频域约束边界，提供更强的监督信号
    """
    
    def __init__(
        self,
        freq_weight: float = 1.0,
        spatial_weight: float = 1.0,
        num_bins: int = 16,
        boundary_width: int = 3,
        high_freq_weight: float = 2.0,
    ):
        super().__init__()
        self.freq_weight = freq_weight
        self.spatial_weight = spatial_weight
        
        # 频率边界损失
        self.freq_boundary_loss = FrequencyAwareBoundaryLoss(
            num_bins=num_bins,
            boundary_width=boundary_width,
            high_freq_weight=high_freq_weight,
        )
        
        # 空域边界损失 (Dice on boundary)
        self.boundary_width = boundary_width
    
    def extract_boundary(self, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """提取边界"""
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        eroded = 1 - F.max_pool2d(1 - mask, kernel_size, stride=1, padding=padding)
        return (dilated - eroded).clamp(0, 1)
    
    def spatial_boundary_dice(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """边界区域 Dice 损失"""
        gt_boundary = self.extract_boundary(gt, self.boundary_width)
        pred_boundary = pred * gt_boundary
        gt_boundary_masked = gt * gt_boundary
        
        intersection = (pred_boundary * gt_boundary_masked).sum(dim=(2, 3))
        union = pred_boundary.sum(dim=(2, 3)) + gt_boundary_masked.sum(dim=(2, 3))
        
        dice = (2 * intersection + 1) / (union + 1)
        return 1 - dice.mean()
    
    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        """计算混合边界损失"""
        if apply_sigmoid:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        
        gt = gt.float()
        
        # 频率边界损失
        freq_loss = self.freq_boundary_loss(pred_prob, gt, apply_sigmoid=False)
        
        # 空域边界损失
        spatial_loss = self.spatial_boundary_dice(pred_prob, gt)
        
        return self.freq_weight * freq_loss + self.spatial_weight * spatial_loss


# ============ 便捷接口 ============

def build_fab_loss(
    num_bins: int = 16,
    boundary_width: int = 3,
    high_freq_weight: float = 2.0,
    use_multiscale: bool = True,
) -> FrequencyAwareBoundaryLoss:
    """构建频率感知边界损失"""
    return FrequencyAwareBoundaryLoss(
        num_bins=num_bins,
        boundary_width=boundary_width,
        high_freq_weight=high_freq_weight,
        use_multiscale=use_multiscale,
    )


def build_hybrid_boundary_loss(
    freq_weight: float = 1.0,
    spatial_weight: float = 1.0,
    num_bins: int = 16,
) -> HybridBoundaryLoss:
    """构建混合边界损失"""
    return HybridBoundaryLoss(
        freq_weight=freq_weight,
        spatial_weight=spatial_weight,
        num_bins=num_bins,
    )


if __name__ == '__main__':
    # 测试
    fab_loss = build_fab_loss()
    hybrid_loss = build_hybrid_boundary_loss()
    
    # 模拟输入
    B = 2
    pred = torch.randn(B, 1, 256, 256)
    gt = torch.zeros(B, 1, 256, 256)
    gt[:, :, 100:110, 100:110] = 1  # 小目标
    
    # 计算损失
    loss1 = fab_loss(pred, gt)
    loss2 = hybrid_loss(pred, gt)
    
    print(f"FAB Loss: {loss1.item():.4f}")
    print(f"Hybrid Boundary Loss: {loss2.item():.4f}")
    print(f"FAB Loss params: {sum(p.numel() for p in fab_loss.parameters()):,}")
