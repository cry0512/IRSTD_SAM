"""
Minimal forward sanity check for EfficientSamHQ.

Runs a single forward pass with random tensors to verify that:
- image encoder returns (image_embeddings, interm_embeddings)
- prompt encoder produces sparse and dense embeddings
- HQ decoder accepts inputs and returns masks and iou predictions
"""

import torch

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_efficient_sam_hq(encoder_patch_embed_dim=192, encoder_num_heads=3).to(device)
    model.eval()

    B = 2
    H = W = 512  # arbitrary input, model preprocess will resize
    Q = 1
    N = 4

    images = torch.randn(B, 3, H, W, device=device)
    with torch.no_grad():
        img_emb, interms = model.get_image_embeddings(images)
    print("image_embeddings:", tuple(img_emb.shape))
    print("interm_embeddings:", tuple(interms.shape))

    # random points in pixel coords [0, W/H), use -1 for padding if needed
    pts = torch.rand(B, Q, N, 2, device=device)
    pts[..., 0] *= W
    pts[..., 1] *= H
    lbl = torch.randint(low=0, high=2, size=(B, Q, N), device=device)

    with torch.no_grad():
        masks, iou = model.predict_masks(
            img_emb,
            interms,
            pts,
            lbl,
            multimask_output=False,
            input_h=H,
            input_w=W,
            output_h=H,
            output_w=W,
            hq_token_only=False,
        )
    print("output masks:", tuple(masks.shape))  # [B, Q, 1, H, W]
    print("iou:", tuple(iou.shape))            # [B, Q, 1]


if __name__ == "__main__":
    main()

