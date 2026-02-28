import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .efficient_sam_prompt_encoder_hq import PromptEncoderHQ
from .two_way_transformer import TwoWayTransformer
from .efficient_sam_encoder_hq import ImageEncoderViTHQ
from .efficient_sam_decoder_hq import MaskDecoderHQ


class SobelDetailEnhancer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(dim, 1, 1, 1))
        self.dim = int(dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.dim)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.dim)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        out = torch.cat([x, magnitude], dim=1)
        return self.fusion(out)


class MultiScaleAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_levels: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_levels)
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * num_levels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, feats):
        if not feats:
            return None
        use_levels = min(len(feats), len(self.convs))
        processed = [self.convs[i](feats[i]) for i in range(use_levels)]
        if len(processed) < len(self.convs):
            # Pad missing levels with zeros to keep fusion shape stable.
            b, c, h, w = processed[0].shape
            for _ in range(len(self.convs) - len(processed)):
                processed.append(torch.zeros((b, c, h, w), device=processed[0].device, dtype=processed[0].dtype))
        out = torch.cat(processed, dim=1)
        return self.fusion_conv(out)


class EfficientSamHQ(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViTHQ,
        prompt_encoder: PromptEncoderHQ,
        mask_decoder: MaskDecoderHQ,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
        use_ms_fusion: bool = False,
        use_detail_enhancer: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False)
        try:
            neck_dim = int(self.image_encoder.neck[0].out_channels)
        except Exception:
            neck_dim = int(getattr(self.image_encoder, "transformer_output_dim", 256))
        mid_dim = max(1, neck_dim // 4)
        self.saliency_adapter = nn.Sequential(
            nn.Conv2d(1, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, neck_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.use_ms_fusion = bool(use_ms_fusion)
        self.use_detail_enhancer = bool(use_detail_enhancer)
        try:
            embed_dim = int(self.image_encoder.patch_embed.proj.out_channels)
        except Exception:
            embed_dim = int(getattr(self.image_encoder, "transformer_output_dim", neck_dim))
        if self.use_detail_enhancer:
            self.detail_enhancer = SobelDetailEnhancer(embed_dim)
        if self.use_ms_fusion:
            self.ms_aggregator = MultiScaleAggregator(in_dim=embed_dim, out_dim=neck_dim)

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> Tuple[torch.Tensor, torch.Tensor]:
        batched_images = self.preprocess(batched_images)
        out = self.image_encoder(batched_images)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            neck_out, interm, ms_feats = out
        else:
            neck_out, interm = out
            ms_feats = []
        if self.use_ms_fusion and ms_feats:
            fused = self.ms_aggregator(ms_feats)
            if fused is not None:
                neck_out = neck_out + fused
        if self.use_detail_enhancer and interm is not None:
            interm_c = interm.permute(0, 3, 1, 2)
            enhanced = self.detail_enhancer(interm_c)
            interm = enhanced.permute(0, 2, 3, 1)
        return neck_out, interm

    def get_image_embeddings_with_text(
        self,
        batched_images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batched_images = self.preprocess(batched_images)
        if not hasattr(self.image_encoder, "forward_with_text"):
            out = self.image_encoder(batched_images)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                neck_out, interm, ms_feats = out
            else:
                neck_out, interm = out
                ms_feats = []
            if self.use_ms_fusion and ms_feats:
                fused = self.ms_aggregator(ms_feats)
                if fused is not None:
                    neck_out = neck_out + fused
            if self.use_detail_enhancer and interm is not None:
                interm_c = interm.permute(0, 3, 1, 2)
                enhanced = self.detail_enhancer(interm_c)
                interm = enhanced.permute(0, 2, 3, 1)
            return neck_out, interm, text_tokens, text_attention_mask
        out = self.image_encoder.forward_with_text(
            batched_images,
            text_tokens,
            text_attention_mask=text_attention_mask,
        )
        if isinstance(out, (tuple, list)) and len(out) == 5:
            neck_out, interm, ms_feats, text_tokens_out, text_mask_out = out
        else:
            neck_out, interm, text_tokens_out, text_mask_out = out
            ms_feats = []
        if self.use_ms_fusion and ms_feats:
            fused = self.ms_aggregator(ms_feats)
            if fused is not None:
                neck_out = neck_out + fused
        if self.use_detail_enhancer and interm is not None:
            interm_c = interm.permute(0, 3, 1, 2)
            enhanced = self.detail_enhancer(interm_c)
            interm = enhanced.permute(0, 2, 3, 1)
        return neck_out, interm, text_tokens_out, text_mask_out

    def apply_saliency_modulation(self, image_embeddings: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        if saliency_map is None:
            return image_embeddings
        target_h, target_w = image_embeddings.shape[-2:]
        saliency_small = F.interpolate(
            saliency_map, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        modulation_weight = self.saliency_adapter(saliency_small)
        return image_embeddings * (1.0 + modulation_weight)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
        hq_token_only: bool = False,
        batched_masks: Optional[torch.Tensor] = None,
        text_sparse_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)
        if batched_masks is not None:
            if batched_masks.dim() == 3:
                batched_masks = batched_masks.unsqueeze(1)
            if batched_masks.shape[0] == batch_size:
                batched_masks = batched_masks.repeat_interleave(max_num_queries, dim=0)
            elif batched_masks.shape[0] != batch_size * max_num_queries:
                raise ValueError("batched_masks must have batch size B or B*Q to match prompts.")
        if text_sparse_embeddings is not None:
            if text_sparse_embeddings.dim() != 3:
                raise ValueError("text_sparse_embeddings must have shape [B, T, C] or [B*Q, T, C].")
            if text_sparse_embeddings.shape[0] == batch_size:
                text_sparse_embeddings = text_sparse_embeddings.repeat_interleave(max_num_queries, dim=0)
            elif text_sparse_embeddings.shape[0] != batch_size * max_num_queries:
                raise ValueError("text_sparse_embeddings must have batch size B or B*Q to match prompts.")

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(
                rescaled_batched_points.reshape(
                    batch_size * max_num_queries, num_pts, 2
                ),
                batched_point_labels.reshape(
                    batch_size * max_num_queries, num_pts
                ),
            ),
            boxes=None,
            masks=batched_masks,
            text_embeds=text_sparse_embeddings,
        )  # sparse: [B*Q, N, C], dense: [B*Q, C, H, W]
        # Repeat along queries
        image_embeddings = image_embeddings.repeat_interleave(max_num_queries, dim=0)
        interm_embeddings = interm_embeddings.repeat_interleave(max_num_queries, dim=0)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
        )

        if output_h > 0 and output_w > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, 1, output_h, output_w),
            )
        else:
            low_res_size = low_res_masks.shape[-1]
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    1,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, 1)
        )
        return output_masks, iou_predictions

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam_hq(
    encoder_patch_embed_dim,
    encoder_num_heads,
    init_from_baseline: Optional[str] = None,
    use_adapter: bool = True,
    use_ms_fusion: bool = False,
    use_detail_enhancer: bool = False,
    early_exit_layer: Optional[int] = None,
):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    prompt_embed_dim = encoder_neck_dims[-1]

    image_encoder = ImageEncoderViTHQ(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type="layer_norm",
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=nn.GELU,
        use_adapter=use_adapter,
        return_multi_scale=use_ms_fusion,
        early_exit_layer=early_exit_layer,
    )

    image_embedding_size = image_encoder.image_embedding_size
    transformer_dim = prompt_embed_dim

    model = EfficientSamHQ(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoderHQ(
            embed_dim=transformer_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            transformer_dim=transformer_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
                activation=nn.GELU,
                normalize_before_activation=False,
            ),
            num_multimask_outputs=1,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_patch_embed_dim,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        use_ms_fusion=use_ms_fusion,
        use_detail_enhancer=use_detail_enhancer,
    )
    # Optional: initialize from baseline EfficientSAM checkpoint (partial, shape-matched)
    if init_from_baseline is not None and os.path.isfile(init_from_baseline):
        try:
            ckpt = torch.load(init_from_baseline, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                src_sd = ckpt["model"]
            else:
                src_sd = ckpt
            dst_sd = model.state_dict()
            loaded, total = 0, 0
            for k, v in src_sd.items():
                if k in dst_sd and isinstance(v, torch.Tensor) and v.shape == dst_sd[k].shape:
                    dst_sd[k] = v
                    loaded += 1
                total += 1
            model.load_state_dict(dst_sd, strict=False)
            print(f"[build_efficient_sam_hq] Partially loaded {loaded} tensors from baseline ({total} scanned).")
        except Exception as e:
            print(f"[build_efficient_sam_hq] Failed to init from baseline: {e}")
    return model
