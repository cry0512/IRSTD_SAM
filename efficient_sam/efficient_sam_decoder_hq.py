from typing import List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBlock


class MaskDecoderHQ(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Upscaling path for SAM branch
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=transformer_dim // 8,
                    num_layers=3,
                    act=activation,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )

        # HQ branch: token + MLP + feature fusion
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 8,
            num_layers=3,
            act=activation,
        )
        # increase token count to include HQ token in the transformer outputs
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,           # [B,C,H,W]
        image_pe: torch.Tensor,                   # [1,C,H,W]
        sparse_prompt_embeddings: torch.Tensor,   # [B, N, C]
        dense_prompt_embeddings: torch.Tensor,    # [B, C, H, W]
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,          # [B, H', W', C] (early ViT grid)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Compose image stream with dense prompt (mask) guidance
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)

        # Upscaling features
        upscaled_embedding_sam = self.output_upscaling(src)

        # HQ features fusion
        vit_features = interm_embeddings.permute(0, 3, 1, 2)  # [B,C',H',W']
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        # Optional radial frequency gate on HQ features
        if getattr(self, "radial_gate", None) is not None:
            try:
                strength = float(getattr(self, "rgate_strength_dec", 0.5))
                hq_features = hq_features + strength * self.radial_gate(hq_features)
            except Exception:
                pass
        # Optional AFD gate on HQ features
        if getattr(self, "afd_gate", None) is not None:
            try:
                afd_strength = float(getattr(self, "afd_strength_dec", 0.5))
                afd_delta = self.afd_gate(hq_features) - hq_features
                hq_features = hq_features + afd_strength * afd_delta
            except Exception:
                pass
        # Optional MSFE gate on HQ features
        if getattr(self, "msfe_gate", None) is not None:
            try:
                msfe_strength = float(getattr(self, "msfe_strength_dec", 0.5))
                msfe_delta = self.msfe_gate(hq_features) - hq_features
                hq_features = hq_features + msfe_strength * msfe_delta
            except Exception:
                pass
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features

        # Hypernets
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                # last token corresponds to HQ token
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_mask_tokens, C']

        b, c2, hh, ww = upscaled_embedding_sam.shape
        # first N-1 are SAM masks, last one is HQ mask
        masks_sam = (hyper_in[:, : self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c2, hh * ww)).view(b, -1, hh, ww)
        masks_hq = (hyper_in[:, self.num_mask_tokens - 1 : self.num_mask_tokens] @ upscaled_embedding_hq.view(b, c2, hh * ww)).view(b, -1, hh, ww)
        masks = torch.cat([masks_sam, masks_hq], dim=1)

        # IoU head
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select outputs
        if multimask_output:
            # choose among multi-mask outputs (exclude first default)
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_sel = iou_pred[:, mask_slice]
            iou_max, max_idx = torch.max(iou_sel, dim=1)
            iou_pred = iou_max.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam_best = masks_multi[torch.arange(masks_multi.size(0)), max_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks_sam_best = masks[:, mask_slice]

        masks_hq_only = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]
        if hq_token_only:
            final_masks = masks_hq_only
        else:
            final_masks = masks_sam_best + masks_hq_only

        return final_masks, iou_pred
