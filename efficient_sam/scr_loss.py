# -*- coding: utf-8 -*-
"""
Signal-to-Clutter Ratio (SCR) Loss for Infrared Small Target Detection

核心创新点:
1. 将红外小目标领域的经典评价指标 SCR 转化为可微分损失
2. 利用原始红外图像信息, 而非仅看 mask 形状
3. 通过环形背景区域的对比度约束, 显式优化目标的可检测性

论文故事:
"传统分割损失 (BCE/Dice) 仅监督 mask 的形状正确性, 忽略了红外图像中
目标与背景的信杂比。我们提出 SCR Loss, 首次将 SCR 指标转化为可微分
监督信号, 在优化分割精度的同时提升目标的可检测性。"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCRLoss(nn.Module):
    """
    Signal-to-Clutter Ratio Loss

    在原始红外图像上计算:
    - 目标区域的平均信号强度 (由预测 soft mask 加权)
    - 环形背景区域的均值和标准差
    - SCR = (signal_mean - bg_mean) / bg_std
    - Loss = 1 / (1 + SCR)  使得 SCR 越大 loss 越小

    Args:
        annular_inner_k: 内环膨胀核大小 (要大于目标)
        annular_outer_k: 外环膨胀核大小 (环形背景的外边界)
        min_target_pixels: 目标最小像素数阈值 (过小时跳过)
        eps: 数值稳定性
    """

    def __init__(
        self,
        annular_inner_k: int = 5,
        annular_outer_k: int = 15,
        min_target_pixels: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.annular_inner_k = annular_inner_k
        self.annular_outer_k = annular_outer_k
        self.min_target_pixels = min_target_pixels
        self.eps = eps

    def _dilate(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """膨胀操作"""
        padding = kernel_size // 2
        return F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

    def forward(
        self,
        pred_logits: torch.Tensor,   # [B, 1, H, W] 预测 logits
        gt_mask: torch.Tensor,        # [B, 1, H, W] GT mask
        images: torch.Tensor,         # [B, C, H, W] 原始红外图像
    ) -> torch.Tensor:
        """
        计算 SCR Loss

        步骤:
        1. 用 GT mask 定义目标区域和环形背景区域
        2. 用预测的 soft mask 在原始图像上提取目标信号
        3. 计算 SCR 并转化为 loss
        """
        pred_prob = torch.sigmoid(pred_logits)     # [B, 1, H, W]
        gt = gt_mask.float()

        # 如果图像是多通道, 取均值转为灰度
        if images.shape[1] > 1:
            gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            gray = images

        B = pred_prob.shape[0]

        # 用 GT 构建环形背景区域 (膨胀GT得到外轮廓, 减去较小的膨胀得到环形)
        inner_dilated = self._dilate(gt, self.annular_inner_k)   # 紧贴目标的区域
        outer_dilated = self._dilate(gt, self.annular_outer_k)   # 更大的区域
        annular_bg = (outer_dilated - inner_dilated).clamp(0, 1)  # 环形背景

        losses = []
        for b in range(B):
            target_mass = gt[b].sum()

            # 跳过没有目标或目标太小的样本
            if target_mass < self.min_target_pixels:
                continue

            bg_mass = annular_bg[b].sum()
            if bg_mass < 1.0:
                continue

            # 目标区域信号: 用预测概率加权的图像平均强度
            target_signal = (pred_prob[b] * gray[b]).sum() / (pred_prob[b].sum() + self.eps)

            # 环形背景统计
            bg_pixels = gray[b] * annular_bg[b]
            bg_mean = bg_pixels.sum() / (bg_mass + self.eps)
            bg_var = ((gray[b] - bg_mean).pow(2) * annular_bg[b]).sum() / (bg_mass + self.eps)
            bg_std = torch.sqrt(bg_var + self.eps)

            # SCR = (target_signal - bg_mean) / bg_std
            scr = (target_signal - bg_mean) / (bg_std + self.eps)

            # Loss: 最大化 SCR → 最小化 1/(1+relu(SCR))
            # 使用 softplus 确保平滑
            loss_b = 1.0 / (1.0 + F.softplus(scr))
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

        return torch.stack(losses).mean()


def build_scr_loss(
    annular_inner_k: int = 5,
    annular_outer_k: int = 15,
    min_target_pixels: float = 4.0,
) -> SCRLoss:
    """构建 SCR Loss"""
    return SCRLoss(
        annular_inner_k=annular_inner_k,
        annular_outer_k=annular_outer_k,
        min_target_pixels=min_target_pixels,
    )


if __name__ == "__main__":
    # 测试
    scr_loss = build_scr_loss()
    B = 2
    pred = torch.randn(B, 1, 256, 256)
    gt = torch.zeros(B, 1, 256, 256)
    gt[:, :, 100:110, 100:110] = 1  # 小目标
    img = torch.rand(B, 3, 256, 256) * 0.3  # 暗背景
    img[:, :, 100:110, 100:110] += 0.5       # 亮目标

    loss = scr_loss(pred, gt, img)
    print(f"SCR Loss: {loss.item():.4f}")
    print(f"SCR Loss params: {sum(p.numel() for p in scr_loss.parameters()):,}")
