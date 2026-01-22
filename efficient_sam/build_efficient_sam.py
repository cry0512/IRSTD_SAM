# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam

def build_efficient_sam_vitt(use_adapter: bool = False, use_ms_fusion: bool = False, use_detail_enhancer: bool = False):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="weights/efficient_sam_vitt.pt",
        use_adapter=use_adapter,
        use_ms_fusion=use_ms_fusion,
        use_detail_enhancer=use_detail_enhancer,
    ).eval()


def build_efficient_sam_vits(use_adapter: bool = False, use_ms_fusion: bool = False, use_detail_enhancer: bool = False):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="weights/efficient_sam_vits.pt",
        use_adapter=use_adapter,
        use_ms_fusion=use_ms_fusion,
        use_detail_enhancer=use_detail_enhancer,
    ).eval()
