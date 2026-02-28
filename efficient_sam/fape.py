# -*- coding: utf-8 -*-
"""
Frequency-Aware Prompt Embedding (FAPE) for IRSTD

核心创新点:
- 在 prompt 点位置提取局部频率特征
- 将频率特征编码到 Prompt Token 中
- 让模型"知道"要分割的目标具有什么样的频率特性

论文故事:
"红外小目标具有独特的频率特性,我们首次将局部频率特征
引入 Prompt 编码,使分割更加精准。"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwarePromptEncoder(nn.Module):
    """
    频率感知的 Prompt 编码器 (FAPE)
    
    在每个 prompt 点位置提取局部频率特征,并与位置编码融合,
    生成频率感知的 Prompt Token。
    
    Args:
        embed_dim: Prompt embedding 维度 (默认 256, 与 SAM 一致)
        freq_patch_size: 局部频率分析的 patch 大小
        num_freq_bins: 径向频率 bin 数量
        use_phase: 是否使用相位特征
        fusion_type: 融合方式 ('add', 'concat', 'attention')
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        freq_patch_size: int = 16,
        num_freq_bins: int = 8,
        use_phase: bool = True,
        fusion_type: str = 'attention',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_patch_size = freq_patch_size
        self.num_freq_bins = num_freq_bins
        self.use_phase = use_phase
        self.fusion_type = fusion_type
        
        # 频率特征维度
        self.freq_feature_dim = num_freq_bins * (2 if use_phase else 1)
        
        # ============ 位置编码 (与 SAM 兼容) ============
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim)
        
        # 点类型编码 (正例/负例)
        self.point_type_embedding = nn.Embedding(2, embed_dim)
        
        # ============ 频率特征编码器 ============
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.freq_feature_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 可学习的频率 bin 重要性权重
        self.freq_bin_importance = nn.Parameter(torch.ones(num_freq_bins))
        
        # ============ 融合模块 ============
        if fusion_type == 'attention':
            self.fusion = FreqSpatialCrossAttention(embed_dim, num_heads=4, dropout=dropout)
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
        else:  # 'add'
            self.fusion = nn.Identity()
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.freq_bin_importance, mean=1.0, std=0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_local_frequency_features(
        self,
        image: torch.Tensor,        # [B, C, H, W] 或 [B, H, W]
        point_coords: torch.Tensor,  # [B, N, 2], 像素坐标
    ) -> torch.Tensor:
        """
        在每个 prompt 点位置提取局部频率特征
        
        Returns:
            freq_features: [B, N, freq_feature_dim]
        """
        if image.dim() == 4:
            # 转为灰度图
            image = image.mean(dim=1)  # [B, H, W]
        
        B, H, W = image.shape
        N = point_coords.shape[1]
        ps = self.freq_patch_size
        half_ps = ps // 2
        
        # 边界安全的坐标
        px = point_coords[..., 0].long().clamp(half_ps, W - half_ps - 1)
        py = point_coords[..., 1].long().clamp(half_ps, H - half_ps - 1)
        
        freq_features_list = []
        
        for b in range(B):
            batch_freq = []
            for n in range(N):
                x, y = px[b, n].item(), py[b, n].item()
                
                # 提取局部 patch
                patch = image[b, y - half_ps : y + half_ps, x - half_ps : x + half_ps]
                
                if patch.shape[0] != ps or patch.shape[1] != ps:
                    # 边界情况: 使用零特征
                    batch_freq.append(torch.zeros(self.freq_feature_dim, device=image.device))
                    continue
                
                # FFT
                fft = torch.fft.rfft2(patch.float(), norm='ortho')
                magnitude = torch.abs(fft)
                
                # 径向 binning
                mag_bins = self._radial_binning(magnitude)
                
                # 应用可学习的重要性权重
                weighted_mag = mag_bins * F.softmax(self.freq_bin_importance, dim=0)
                
                if self.use_phase:
                    phase = torch.angle(fft)
                    phase_bins = self._radial_binning(phase)
                    freq_feat = torch.cat([weighted_mag, phase_bins], dim=0)
                else:
                    freq_feat = weighted_mag
                
                batch_freq.append(freq_feat)
            
            freq_features_list.append(torch.stack(batch_freq, dim=0))
        
        return torch.stack(freq_features_list, dim=0)  # [B, N, freq_dim]
    
    def _radial_binning(self, freq_map: torch.Tensor) -> torch.Tensor:
        """将 2D 频谱按径向距离分 bin 并平均"""
        h, w = freq_map.shape
        device = freq_map.device
        
        # 频率坐标 - 使用实际的频谱尺寸
        # rfft2 输出尺寸: (h, w//2+1)，所以 w 已经是压缩后的宽度
        fy = torch.fft.fftfreq(h, device=device)
        fx = torch.linspace(0, 0.5, w, device=device)  # 直接生成匹配尺寸的频率
        
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
        radius = torch.sqrt(fx_grid**2 + fy_grid**2)
        
        # 分 bin
        max_r = radius.max().item() + 1e-6
        bin_edges = torch.linspace(0, max_r, self.num_freq_bins + 1, device=device)
        
        bins = torch.zeros(self.num_freq_bins, device=device)
        for i in range(self.num_freq_bins):
            mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
            if mask.sum() > 0:
                bins[i] = freq_map[mask].mean()
        
        return bins
    
    def forward(
        self,
        image: torch.Tensor,         # [B, C, H, W]
        point_coords: torch.Tensor,  # [B, N, 2], 像素坐标
        point_labels: torch.Tensor,  # [B, N], 0=负例, 1=正例
    ) -> torch.Tensor:
        """
        生成频率感知的 Prompt Embedding
        
        Returns:
            prompt_embedding: [B, N, embed_dim]
        """
        B, N, _ = point_coords.shape
        _, _, H, W = image.shape
        
        # ============ 空间分支: 位置 + 类型编码 ============
        # 归一化坐标
        norm_coords = point_coords.clone().float()
        norm_coords[..., 0] = norm_coords[..., 0] / W
        norm_coords[..., 1] = norm_coords[..., 1] / H
        
        pos_embed = self.positional_encoding(norm_coords)  # [B, N, embed_dim]
        type_embed = self.point_type_embedding(point_labels.long())  # [B, N, embed_dim]
        spatial_embed = pos_embed + type_embed
        
        # ============ 频率分支: 局部频率特征 ============
        freq_features = self.extract_local_frequency_features(image, point_coords)
        freq_embed = self.freq_encoder(freq_features)  # [B, N, embed_dim]
        
        # ============ 融合 ============
        if self.fusion_type == 'attention':
            fused_embed = self.fusion(spatial_embed, freq_embed)
        elif self.fusion_type == 'concat':
            concat_embed = torch.cat([spatial_embed, freq_embed], dim=-1)
            fused_embed = self.fusion(concat_embed)
        else:  # 'add'
            fused_embed = spatial_embed + freq_embed
        
        # 输出投影
        output = self.output_proj(fused_embed)
        
        return output


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 (与 SAM 兼容)"""
    
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


class FreqSpatialCrossAttention(nn.Module):
    """空间-频率交叉注意力融合"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, spatial_embed: torch.Tensor, freq_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_embed: [B, N, D] - Query
            freq_embed: [B, N, D] - Key, Value
        Returns:
            fused: [B, N, D]
        """
        # Cross-attention: spatial attends to frequency
        attn_out, _ = self.cross_attn(spatial_embed, freq_embed, freq_embed)
        x = self.norm1(spatial_embed + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


# ============ 便捷接口 ============

def build_fape_encoder(
    embed_dim: int = 256,
    freq_patch_size: int = 16,
    num_freq_bins: int = 8,
    fusion_type: str = 'attention',
) -> FrequencyAwarePromptEncoder:
    """构建 FAPE 编码器"""
    return FrequencyAwarePromptEncoder(
        embed_dim=embed_dim,
        freq_patch_size=freq_patch_size,
        num_freq_bins=num_freq_bins,
        use_phase=True,
        fusion_type=fusion_type,
    )


if __name__ == '__main__':
    # 测试
    fape = build_fape_encoder()
    
    # 模拟输入
    image = torch.randn(2, 3, 256, 256)
    point_coords = torch.randint(32, 224, (2, 5, 2)).float()
    point_labels = torch.randint(0, 2, (2, 5))
    
    # 前向传播
    output = fape(image, point_coords, point_labels)
    print(f"FAPE output shape: {output.shape}")  # [2, 5, 256]
    print(f"FAPE parameters: {sum(p.numel() for p in fape.parameters()):,}")
