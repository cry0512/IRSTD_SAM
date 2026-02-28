# -*- coding: utf-8 -*-
"""
Contrastive Prompt Learning (CPL) for IRSTD

核心创新点:
- 利用对比学习增强 Prompt Embedding 的区分性
- 正例点 (目标上) 在特征空间中聚类
- 负例点 (背景上) 被推远
- 与 RadialGate/ASG2 等 Encoder 模块协同工作

论文故事:
"我们提出对比学习驱动的 Prompt 学习策略,
通过拉近目标区域的 prompt 表征、推远背景表征,
显著提升小目标分割的精度。"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastivePromptLearning(nn.Module):
    """
    对比学习增强的 Prompt 编码模块
    
    可与任何 Prompt Encoder (FAPE, SFJP) 或直接与 SAM 默认 Prompt Encoder 配合使用。
    也可以独立使用，直接对 SAM 的点 prompt 进行对比学习增强。
    
    Args:
        embed_dim: Prompt embedding 维度 (SAM 默认 256)
        proj_dim: 对比学习投影空间维度
        temperature: 对比学习温度参数
        use_hard_negatives: 是否使用困难负例挖掘
        margin: 三元组损失的 margin (如果使用)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        loss_type: str = 'infonce',  # 'infonce', 'triplet', 'ntxent'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.loss_type = loss_type
        
        # ============ Prompt Embedding (与 SAM 兼容) ============
        # 位置编码
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim)
        
        # 点类型编码 (正例/负例)
        self.point_type_embedding = nn.Embedding(2, embed_dim)
        
        # Prompt 特征增强 (轻量级 MLP)
        self.prompt_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # ============ 对比学习投影头 ============
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        
        # 可学习的温度参数
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode_prompts(
        self,
        point_coords: torch.Tensor,  # [B, N, 2], 像素坐标
        point_labels: torch.Tensor,  # [B, N], 0=负例, 1=正例, -1=忽略
        image_size: tuple = (256, 256),
    ) -> torch.Tensor:
        """
        编码 prompt 点为 embedding
        
        Returns:
            prompt_embed: [B, N, embed_dim]
        """
        B, N, _ = point_coords.shape
        H, W = image_size
        
        # 归一化坐标
        norm_coords = point_coords.clone().float()
        norm_coords[..., 0] = norm_coords[..., 0] / W
        norm_coords[..., 1] = norm_coords[..., 1] / H
        
        # 位置编码
        pos_embed = self.pos_encoder(norm_coords)  # [B, N, embed_dim]
        
        # 类型编码 - clamp to valid range [0, 1] to avoid index errors
        labels_clamped = point_labels.long().clamp(0, 1)
        type_embed = self.point_type_embedding(labels_clamped)  # [B, N, embed_dim]
        
        # 组合
        prompt_embed = pos_embed + type_embed
        
        # 增强
        prompt_embed = self.prompt_enhancer(prompt_embed)
        
        return prompt_embed
    
    def compute_contrastive_loss(
        self,
        prompt_embed: torch.Tensor,  # [B, N, embed_dim]
        point_labels: torch.Tensor,  # [B, N], 0=负例, 1=正例, -1=忽略
    ) -> torch.Tensor:
        """
        计算对比学习损失
        
        正例: point_labels == 1 的点
        负例: point_labels == 0 的点
        忽略: point_labels == -1 的点
        
        目标: 正例之间拉近, 正负例之间推远
        """
        B, N, D = prompt_embed.shape
        device = prompt_embed.device
        
        # 投影到对比学习空间
        proj_embed = self.projector(prompt_embed)  # [B, N, proj_dim]
        proj_embed = F.normalize(proj_embed, dim=-1)  # L2 归一化
        
        # 获取温度
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)
        
        total_loss = 0.0
        valid_count = 0
        
        for b in range(B):
            # 使用 CPU 上的 bool 索引来避免 CUDA 错误
            labels_b = point_labels[b].cpu()
            pos_indices = (labels_b == 1).nonzero(as_tuple=True)[0]
            neg_indices = (labels_b == 0).nonzero(as_tuple=True)[0]
            
            n_pos = len(pos_indices)
            n_neg = len(neg_indices)
            
            if n_pos < 2 or n_neg < 1:
                continue  # 需要至少2个正例和1个负例
            
            # 使用索引获取 embeddings
            pos_embeds = proj_embed[b][pos_indices.to(device)]  # [N_pos, proj_dim]
            neg_embeds = proj_embed[b][neg_indices.to(device)]  # [N_neg, proj_dim]
            
            if self.loss_type == 'infonce':
                loss = self._infonce_loss(pos_embeds, neg_embeds, temperature)
            elif self.loss_type == 'ntxent':
                loss = self._ntxent_loss(pos_embeds, neg_embeds, temperature)
            elif self.loss_type == 'triplet':
                loss = self._triplet_loss(pos_embeds, neg_embeds)
            else:
                loss = self._infonce_loss(pos_embeds, neg_embeds, temperature)
            
            total_loss += loss
            valid_count += 1
        
        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / valid_count
    
    def _infonce_loss(self, pos_embeds, neg_embeds, temperature):
        """InfoNCE 损失"""
        n_pos = pos_embeds.shape[0]
        loss = 0.0
        
        for i in range(n_pos):
            anchor = pos_embeds[i]  # [proj_dim]
            
            # 正例相似度 (排除自己)
            pos_sim = torch.matmul(pos_embeds, anchor)  # [N_pos]
            pos_sim_masked = pos_sim.clone()
            pos_sim_masked[i] = -1e9  # mask 自己
            
            # 负例相似度
            neg_sim = torch.matmul(neg_embeds, anchor)  # [N_neg]
            
            # 困难负例挖掘
            if self.use_hard_negatives:
                # 选择最相似的负例
                hard_neg_sim = neg_sim.max()
                # 组合: 最难的负例 vs 其他正例
                logits = torch.cat([pos_sim_masked, neg_sim], dim=0) / temperature
            else:
                logits = torch.cat([pos_sim_masked, neg_sim], dim=0) / temperature
            
            # 第一个正例作为正样本 (实际上是最相似的正例)
            target_idx = pos_sim_masked.argmax()
            loss += F.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
        
        return loss / n_pos
    
    def _ntxent_loss(self, pos_embeds, neg_embeds, temperature):
        """NT-Xent 损失 (SimCLR 风格)"""
        n_pos = pos_embeds.shape[0]
        
        # 正例之间的相似度矩阵
        pos_sim = torch.matmul(pos_embeds, pos_embeds.T) / temperature  # [N_pos, N_pos]
        
        # 正例与负例的相似度
        neg_sim = torch.matmul(pos_embeds, neg_embeds.T) / temperature  # [N_pos, N_neg]
        
        # 对角线 mask (排除自己)
        mask = torch.eye(n_pos, dtype=torch.bool, device=pos_embeds.device)
        pos_sim = pos_sim.masked_fill(mask, -1e9)
        
        # 合并 logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [N_pos, N_pos + N_neg]
        
        # 对每个 anchor，正例在前 N_pos-1 个位置
        labels = torch.arange(n_pos, device=pos_embeds.device)
        labels = (labels + 1) % n_pos  # 循环选择下一个正例作为 target
        
        return F.cross_entropy(logits, labels)
    
    def _triplet_loss(self, pos_embeds, neg_embeds, margin=0.5):
        """三元组损失"""
        n_pos = pos_embeds.shape[0]
        loss = 0.0
        
        for i in range(n_pos):
            anchor = pos_embeds[i]
            
            # 最难的正例 (最远的正例)
            pos_dist = 1 - torch.matmul(pos_embeds, anchor)
            pos_dist[i] = -1e9
            hard_pos_dist = pos_dist.max()
            
            # 最难的负例 (最近的负例)
            neg_dist = 1 - torch.matmul(neg_embeds, anchor)
            hard_neg_dist = neg_dist.min()
            
            # Triplet loss
            loss += F.relu(hard_pos_dist - hard_neg_dist + margin)
        
        return loss / n_pos
    
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_size: tuple = (256, 256),
        return_loss: bool = True,
    ) -> tuple:
        """
        前向传播
        
        Args:
            point_coords: [B, N, 2] 像素坐标
            point_labels: [B, N] 点标签 (0=负例, 1=正例)
            image_size: 图像尺寸 (H, W)
            return_loss: 是否返回对比损失
            
        Returns:
            prompt_embed: [B, N, embed_dim]
            cl_loss: 对比学习损失 (如果 return_loss=True)
        """
        # 编码 prompts
        prompt_embed = self.encode_prompts(point_coords, point_labels, image_size)
        
        if return_loss and self.training:
            cl_loss = self.compute_contrastive_loss(prompt_embed, point_labels)
            return prompt_embed, cl_loss
        
        return prompt_embed, torch.tensor(0.0, device=prompt_embed.device)


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
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

def build_contrastive_prompt_learning(
    embed_dim: int = 256,
    proj_dim: int = 128,
    temperature: float = 0.07,
    loss_type: str = 'infonce',
) -> ContrastivePromptLearning:
    """构建对比学习 Prompt 模块"""
    return ContrastivePromptLearning(
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        temperature=temperature,
        use_hard_negatives=True,
        loss_type=loss_type,
    )


if __name__ == '__main__':
    # 测试
    cpl = build_contrastive_prompt_learning()
    
    # 模拟输入
    B, N = 2, 8  # 2个样本，每个8个点
    point_coords = torch.randint(32, 224, (B, N, 2)).float()
    point_labels = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],  # 4正4负
        [1, 1, 1, 0, 0, 0, 0, 0],  # 3正5负
    ])
    
    # 训练模式
    cpl.train()
    prompt_embed, cl_loss = cpl(point_coords, point_labels, image_size=(256, 256))
    print(f"Prompt embedding shape: {prompt_embed.shape}")
    print(f"Contrastive loss: {cl_loss.item():.4f}")
    print(f"CPL parameters: {sum(p.numel() for p in cpl.parameters()):,}")
