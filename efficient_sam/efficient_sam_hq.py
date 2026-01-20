import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .efficient_sam_prompt_encoder_hq import PromptEncoderHQ
from .two_way_transformer import TwoWayTransformer
from .efficient_sam_encoder_hq import ImageEncoderViTHQ
from .efficient_sam_decoder_hq import MaskDecoderHQ


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
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False)

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> Tuple[torch.Tensor, torch.Tensor]:
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

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
            masks=None,
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


def build_efficient_sam_hq(encoder_patch_embed_dim, encoder_num_heads, init_from_baseline: Optional[str] = None):
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
