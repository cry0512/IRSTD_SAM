# -*- coding: utf-8 -*-
"""
Spatial-Frequency Joint Prompt (SFJP) for IRSTD

核心创新点:
- 双分支设计: 空间域分支 + 频率域分支
- 在图像级别提取全局空间和频率特征
- 使用交叉注意力融合双域信息生成 Prompt

与 FAPE 的区别:
- FAPE: 在 prompt 点位置提取 **局部** 频率特征
- SFJP: 提取 **全局** 空间和频率特征, 然后采样到 prompt 位置

论文故事:
"我们提出空间-频率双域联合的 Prompt 编码方案,
同时捕捉目标的空间位置信息和频率特性。"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyJointPromptEncoder(nn.Module):
    """
    空间-频率联合 Prompt 编码器 (SFJP)
    
    双分支架构:
    1. Spatial Branch: 从图像提取空间特征, 在 prompt 位置采样
    2. Frequency Branch: 从图像的频谱提取特征, 在 prompt 位置采样
    3. Fusion Module: 交叉注意力融合双域信息
    
    Args:
        embed_dim: Prompt embedding 维度
        spatial_channels: 空间分支的通道数
        freq_channels: 频率分支的通道数
        num_heads: 注意力头数
        use_magnitude: 是否使用幅度谱
        use_phase: 是否使用相位谱
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        spatial_channels: int = 64,
        freq_channels: int = 64,
        num_heads: int = 4,
        use_magnitude: bool = True,
        use_phase: bool = True,
        dropout: float = 0.1,
        in_channels: int = 1,  # 默认1通道 (适配SCTransNet灰度预处理)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase
        self.in_channels = in_channels
        
        # ============ Spatial Branch ============
        # 轻量级空间特征提取 (不需要太深, 主要是提供位置信息)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels // 2),
            nn.GELU(),
            nn.Conv2d(spatial_channels // 2, spatial_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(spatial_channels),
            nn.GELU(),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
        )
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # ============ Frequency Branch ============
        # 频谱特征通道数
        freq_input_channels = 0
        if use_magnitude:
            freq_input_channels += 1
        if use_phase:
            freq_input_channels += 1
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(freq_input_channels, freq_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(freq_channels // 2),
            nn.GELU(),
            nn.Conv2d(freq_channels // 2, freq_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(freq_channels),
            nn.GELU(),
            nn.Conv2d(freq_channels, freq_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(freq_channels),
        )
        
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # ============ Positional Encoding ============
        self.pos_encoding = SinusoidalPositionalEncoding2D(embed_dim)
        
        # 点类型编码
        self.point_type_embedding = nn.Embedding(2, embed_dim)
        
        # ============ Cross-Attention Fusion ============
        self.cross_attn_s2f = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_f2s = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 融合 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 可学习的域权重 (自适应调整空间/频率的重要性)
        self.domain_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def extract_frequency_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        提取图像的频率域特征
        
        Args:
            image: [B, C, H, W]
        Returns:
            freq_features: [B, freq_channels, H, W]
        """
        B, C, H, W = image.shape
        
        # 转灰度图 (如果输入已经是1通道则直接使用)
        if C == 1:
            gray = image
        else:
            gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 保存原始 dtype 并转换为 float32 以避免 ComplexHalf 问题
        orig_dtype = gray.dtype
        gray_fp32 = gray.float() if orig_dtype == torch.float16 else gray
        
        # 2D FFT
        fft = torch.fft.rfft2(gray_fp32, norm='ortho')  # [B, 1, H, W//2+1]
        
        freq_features = []
        
        if self.use_magnitude:
            magnitude = torch.abs(fft)
            # 对数压缩 (避免极端值)
            magnitude = torch.log1p(magnitude)
            # 上采样回原始尺寸 (rfft 输出宽度减半)
            magnitude = F.interpolate(magnitude, size=(H, W), mode='bilinear', align_corners=False)
            freq_features.append(magnitude)
        
        if self.use_phase:
            phase = torch.angle(fft)
            # 归一化到 [0, 1]
            phase = (phase + math.pi) / (2 * math.pi)
            phase = F.interpolate(phase, size=(H, W), mode='bilinear', align_corners=False)
            freq_features.append(phase)
        
        freq_input = torch.cat(freq_features, dim=1)  # [B, 1 or 2, H, W]
        
        # 恢复 dtype
        if orig_dtype == torch.float16:
            freq_input = freq_input.half()
        
        return freq_input
    
    def sample_at_points(
        self,
        feature_map: torch.Tensor,  # [B, C, H, W]
        point_coords: torch.Tensor,  # [B, N, 2], 像素坐标
    ) -> torch.Tensor:
        """
        在 prompt 点位置采样特征
        
        Returns:
            sampled: [B, N, C]
        """
        B, C, H, W = feature_map.shape
        N = point_coords.shape[1]
        
        # 归一化坐标到 [-1, 1] (grid_sample 需要)
        grid = point_coords.clone().float()
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        
        # [B, N, 2] -> [B, 1, N, 2]
        grid = grid.unsqueeze(1)
        
        # grid_sample
        sampled = F.grid_sample(
            feature_map, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # [B, C, 1, N]
        
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]
        
        return sampled
    
    def forward(
        self,
        image: torch.Tensor,         # [B, C, H, W]
        point_coords: torch.Tensor,  # [B, N, 2], 像素坐标
        point_labels: torch.Tensor,  # [B, N], 0=负例, 1=正例
    ) -> torch.Tensor:
        """
        生成空间-频率联合的 Prompt Embedding
        
        Returns:
            prompt_embedding: [B, N, embed_dim]
        """
        B, C, H, W = image.shape
        N = point_coords.shape[1]
        
        # ============ Spatial Branch ============
        # 自动适配输入通道数 (支持1通道或3通道)
        if C != self.in_channels:
            if C == 1 and self.in_channels == 3:
                # 灰度图扩展为3通道
                image_spatial = image.expand(-1, 3, -1, -1)
            elif C == 3 and self.in_channels == 1:
                # RGB转灰度
                image_spatial = image.mean(dim=1, keepdim=True)
            else:
                image_spatial = image
        else:
            image_spatial = image
        
        spatial_feat_map = self.spatial_encoder(image_spatial)  # [B, spatial_ch, H/2, W/2]
        
        # 坐标转换 (因为 stride=2)
        scaled_coords = point_coords.clone().float()
        scaled_coords = scaled_coords / 2.0
        
        spatial_feat = self.sample_at_points(spatial_feat_map, scaled_coords)  # [B, N, spatial_ch]
        spatial_embed = self.spatial_proj(spatial_feat)  # [B, N, embed_dim]
        
        # ============ Frequency Branch ============
        freq_input = self.extract_frequency_features(image)  # [B, 1 or 2, H, W]
        freq_feat_map = self.freq_encoder(freq_input)  # [B, freq_ch, H, W]
        
        freq_feat = self.sample_at_points(freq_feat_map, point_coords)  # [B, N, freq_ch]
        freq_embed = self.freq_proj(freq_feat)  # [B, N, embed_dim]
        
        # ============ Positional Encoding ============
        norm_coords = point_coords.clone().float()
        norm_coords[..., 0] = norm_coords[..., 0] / W
        norm_coords[..., 1] = norm_coords[..., 1] / H
        
        pos_embed = self.pos_encoding(norm_coords)  # [B, N, embed_dim]
        type_embed = self.point_type_embedding(point_labels.long())  # [B, N, embed_dim]
        
        # 添加位置和类型信息
        spatial_embed = spatial_embed + pos_embed + type_embed
        freq_embed = freq_embed + pos_embed + type_embed
        
        # ============ Cross-Attention Fusion ============
        # Spatial attends to Frequency
        s2f, _ = self.cross_attn_s2f(spatial_embed, freq_embed, freq_embed)
        s2f = self.norm1(spatial_embed + s2f)
        
        # Frequency attends to Spatial
        f2s, _ = self.cross_attn_f2s(freq_embed, spatial_embed, spatial_embed)
        f2s = self.norm2(freq_embed + f2s)
        
        # 自适应权重融合
        weights = F.softmax(self.domain_weight, dim=0)
        weighted_fusion = weights[0] * s2f + weights[1] * f2s
        
        # MLP 融合
        concat_embed = torch.cat([s2f, f2s], dim=-1)  # [B, N, 2*embed_dim]
        output = self.fusion_mlp(concat_embed)  # [B, N, embed_dim]
        
        # 残差连接
        output = output + weighted_fusion
        
        return output


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D 正弦位置编码"""
    
    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 2], 归一化坐标 [0, 1]
        Returns:
            embeddings: [B, N, embed_dim]
        """
        device = coords.device
        dim_t = torch.arange(self.embed_dim // 4, device=device).float()
        dim_t = self.temperature ** (2 * dim_t / (self.embed_dim // 4))
        
        x = coords[..., 0:1] * 2 * math.pi
        y = coords[..., 1:2] * 2 * math.pi
        
        pe_x = torch.cat([torch.sin(x / dim_t), torch.cos(x / dim_t)], dim=-1)
        pe_y = torch.cat([torch.sin(y / dim_t), torch.cos(y / dim_t)], dim=-1)
        
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return self.proj(pe)


# ============ 便捷接口 ============

def build_sfjp_encoder(
    embed_dim: int = 256,
    spatial_channels: int = 64,
    freq_channels: int = 64,
    num_heads: int = 4,
) -> SpatialFrequencyJointPromptEncoder:
    """构建 SFJP 编码器"""
    return SpatialFrequencyJointPromptEncoder(
        embed_dim=embed_dim,
        spatial_channels=spatial_channels,
        freq_channels=freq_channels,
        num_heads=num_heads,
        use_magnitude=True,
        use_phase=True,
    )


if __name__ == '__main__':
    # 测试
    sfjp = build_sfjp_encoder()
    
    # 模拟输入
    image = torch.randn(2, 3, 256, 256)
    point_coords = torch.randint(32, 224, (2, 5, 2)).float()
    point_labels = torch.randint(0, 2, (2, 5))
    
    # 前向传播
    output = sfjp(image, point_coords, point_labels)
    print(f"SFJP output shape: {output.shape}")  # [2, 5, 256]
    print(f"SFJP parameters: {sum(p.numel() for p in sfjp.parameters()):,}")
    
    # 打印域权重
    print(f"Domain weights: spatial={F.softmax(sfjp.domain_weight, dim=0)[0]:.3f}, "
          f"freq={F.softmax(sfjp.domain_weight, dim=0)[1]:.3f}")
