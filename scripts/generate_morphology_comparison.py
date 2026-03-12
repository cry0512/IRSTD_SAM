import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_text_sparse_prompt_projector,
    build_text_dense_mask_prompt_generator_v2,
    build_text_dense_mask_prompt_generator,
    build_gated_backbone_bifusion_block_adapter,
)

# Hardcoded absolute paths to the two checkpoints
CKPT_BASELINE = r'\\?\E:\code\EfficientSAM-main\EfficientSAM-main\outputs_sam_sirst_hq\NUAA-SIRST_model-vitt_size256_HQ_baseline_bs4_e1000_kp1msfx_pixels0_split-50_50_20260203_104435\best_ep099_miou71p85_niou69p83_f183p62_pd98p10_fa22p46.pt'
CKPT_ASG = r'\\?\E:\code\EfficientSAM-main\EfficientSAM-main\outputs_sam_sirst_hq\NUAA-SIRST_model-vitt_size256_use_HQ_asg2_SCPreproc_bs4_e1000_kp1_pixels0_split-50_50_20260210_093038\best_ep640_miou81p33_niou81p33_f183p58_pd96p40_fa22p03.pt'

class DictArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def build_model_from_ckpt(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if 'args' not in ckpt:
        # Fallback to tmp json if baseline lacked embedded args
        with open(r'E:\code\EfficientSAM-main\EfficientSAM-main\tmp_model_args.json', 'r') as f:
            args_dict = json.load(f)
    else:
        args_dict = ckpt['args']
        
    args = DictArgs(**args_dict)
    
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192 if args.model == "vitt" else 384,
        encoder_num_heads=3 if args.model == "vitt" else 6,
        init_from_baseline=None,
        use_adapter=getattr(args, 'use_fs_adapter', False),
        use_ms_fusion=getattr(args, 'use_ms_fusion', False),
        use_detail_enhancer=getattr(args, 'use_detail_enhancer', False),
        early_exit_layer=getattr(args, 'early_exit_layer', 0),
    ).to(device)

    # Attach frequency gates if requested
    if getattr(args, 'use_asg_hq', False):
        from efficient_sam.asg import AnisotropicSpectralGating, AnisotropicSpectralGating2
        class _ASGDelta(nn.Module):
            def __init__(self, asg):
                super().__init__()
                self.asg = asg
            def forward(self, x):
                return self.asg(x) - x

        asg_variant = getattr(args, 'asg_variant', 'asg1')
        if asg_variant == "asg2":
            asg_cls = AnisotropicSpectralGating2
            asg_kwargs = {"r_bins": getattr(args, 'asg_radial_bins', 64), "theta_bins": getattr(args, 'asg_angular_bins', 128)}
        else:
            asg_cls = AnisotropicSpectralGating
            asg_kwargs = {"num_radial_bins": getattr(args, 'asg_radial_bins', 64), "num_angular_bins": getattr(args, 'asg_angular_bins', 128)}

        enc_hw = getattr(model.image_encoder, "image_embedding_size", 64)
        asg_loc = getattr(args, 'asg_loc', 'encoder')
        if asg_loc in ("encoder", "both"):
            dim_enc = model.image_encoder.neck[0].out_channels
            asg_enc = asg_cls(dim_enc, enc_hw, enc_hw, **asg_kwargs).to(device)
            model.image_encoder.radial_gate = _ASGDelta(asg_enc)
            model.image_encoder.rgate_strength = float(getattr(args, 'asg_strength_enc', 1.0))
        if asg_loc in ("decoder", "both"):
            c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
            hq_hw = enc_hw * 4
            asg_dec = asg_cls(c_dec, hq_hw, hq_hw, **asg_kwargs).to(device)
            model.mask_decoder.radial_gate = _ASGDelta(asg_dec)
            model.mask_decoder.rgate_strength_dec = float(getattr(args, 'asg_strength_dec', 1.0))

    img_dim = 256
    if getattr(args, "use_text_sparse_prompt", False):
        sparse_projector = build_text_sparse_prompt_projector(
            text_dim=args.mllm_text_dim,
            embed_dim=getattr(model.prompt_encoder, "embed_dim", 256),
            num_tokens=max(1, int(args.text_sparse_num_tokens)),
            init_scale=float(args.text_sparse_init_scale),
            use_raw_global_gate=bool(getattr(args, "text_sparse_raw_global_gate", False)),
            raw_global_gate_init_bias=float(getattr(args, "text_sparse_raw_global_gate_init_bias", -2.0)),
        ).to(device)
    else:
        sparse_projector = None

    if getattr(args, "use_text_dense_prompt", False):
        dense_variant = getattr(args, "text_dense_prompt_type", "global")
        if dense_variant == "token_xattn":
            dense_generator = build_text_dense_mask_prompt_generator_v2(
                img_dim=img_dim, text_dim=args.mllm_text_dim,
                hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                num_heads=max(1, int(args.text_dense_num_heads)),
            ).to(device)
        else:
            dense_generator = build_text_dense_mask_prompt_generator(
                img_dim=img_dim, text_dim=args.mllm_text_dim,
                hidden_dim=max(8, int(args.text_dense_hidden_dim)),
            ).to(device)
    else:
        dense_generator = None

    if getattr(args, "use_gated_bifusion_backbone_blocks", False):
        vision_dim = int(model.image_encoder.patch_embed.proj.out_channels)
        num_layers = len(getattr(model.image_encoder, "blocks", []))
        common_kwargs = dict(
            num_layers=max(1, int(num_layers)), vision_dim=vision_dim, text_dim=args.mllm_text_dim,
            hidden_dim=max(8, int(getattr(args, "bifusion_hidden_dim", 128))),
            num_heads=max(1, int(getattr(args, "bifusion_num_heads", 4))),
            apply_every=max(1, int(getattr(args, "bifusion_block_apply_every", 1))),
            vision_res_scale=float(getattr(args, "bifusion_block_vision_res_scale", 1.0)),
            text_res_scale=float(getattr(args, "bifusion_block_text_res_scale", 1.0)),
        )
        bifusion_backbone = build_gated_backbone_bifusion_block_adapter(
            gate_hidden_dim=int(getattr(args, "bifusion_gate_hidden_dim", 0)),
            gate_init_bias=float(getattr(args, "bifusion_gate_init_bias", -2.0)),
            **common_kwargs,
        ).to(device)
        model.image_encoder.bifusion_adapter = bifusion_backbone
    
    model.text_sparse_projector = sparse_projector
    model.text_dense_generator = dense_generator

    sd = ckpt['model']
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, args

def inference(model, args, batch, device):
    img = batch["image"].to(device)
    mask_label = batch["mask"].squeeze(0).squeeze(0).numpy() # Shape [H, W]

    clip_feat = batch.get("clip_text_feat", None)
    clip_token_feat = batch.get("clip_text_token_feat", None)
    clip_token_mask = batch.get("clip_text_attn_mask", None)
    
    if clip_feat is not None: clip_feat = clip_feat.to(device)
    if clip_token_feat is not None: clip_token_feat = clip_token_feat.to(device)
    if clip_token_mask is not None: clip_token_mask = clip_token_mask.to(device)
    
    b, c, h, w = img.shape
    y_idx, x_idx = np.where(mask_label > 0)
    if len(y_idx) > 0:
        cx, cy = float(np.mean(x_idx)), float(np.mean(y_idx))
    else:
        cx, cy = w / 2.0, h / 2.0
        
    pts = torch.tensor([[[[cx, cy]]]], dtype=torch.float32, device=device)
    pts_labels = torch.tensor([[[1]]], dtype=torch.float32, device=device)
    
    if getattr(args, 'use_gated_bifusion_backbone_blocks', False) and clip_feat is not None:
        text_tokens = clip_token_feat if clip_token_feat is not None else clip_feat.unsqueeze(1)
        t_mask = clip_token_mask if clip_token_mask is not None else torch.ones((text_tokens.shape[0], 1), device=device, dtype=torch.long)
        out = model.get_image_embeddings_with_text(img, text_tokens, text_attention_mask=t_mask)
        if len(out) == 4:
            img_emb, interms, _, _ = out
        else:
            img_emb, interms = out[0], out[1]
    else:
        img_emb, interms = model.get_image_embeddings(img)
        
    sparse_p = None
    if model.text_sparse_projector is not None and clip_feat is not None:
        source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
        sparse_input = clip_token_feat if source == "fused_tokens" else clip_feat
        sparse_mask = clip_token_mask if source == "fused_tokens" else None
        
        if sparse_input is not None:
            if sparse_input.dim() == 3:
                sparse_p = model.text_sparse_projector(sparse_input, attention_mask=sparse_mask)
            else:
                sparse_p = model.text_sparse_projector(sparse_input, use_global_prompt_enhance=(source == "raw_global"))
    
    dense_p = None
    if model.text_dense_generator is not None and clip_feat is not None:
        target_size = getattr(model.prompt_encoder, "mask_input_size", None)
        if getattr(model.text_dense_generator, "expects_token_level", False):
            dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
            if dense_text_input is not None:
                dense_p = model.text_dense_generator(
                    img_emb, dense_text_input,
                    attention_mask=clip_token_mask if clip_token_feat is not None else None,
                    output_size=tuple(target_size) if target_size is not None else None,
                )
        else:
            dense_p = model.text_dense_generator(
                img_emb, clip_feat, output_size=tuple(target_size) if target_size is not None else None
            )
        if dense_p is not None:
            dense_p = dense_p * float(getattr(args, "text_dense_prompt_scale", 1.0))
            
    input_h, input_w = int(batch.get("sizes", [[256, 256]])[0][0]), int(batch.get("sizes", [[256, 256]])[0][1])

    predicted_logits, _ = model.predict_masks(
        img_emb, interms, pts, pts_labels, 
        multimask_output=False,
        input_h=input_h, input_w=input_w,
        output_h=input_h, output_w=input_w,
        hq_token_only=False,
        batched_masks=dense_p,
        text_sparse_embeddings=sparse_p
    )
    
    logits = predicted_logits.squeeze() # Shape [H_feat, W_feat]
    prob_map = torch.sigmoid(logits)
    return prob_map

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_baseline, args_baseline = build_model_from_ckpt(CKPT_BASELINE, device)
    model_asg, args_asg = build_model_from_ckpt(CKPT_ASG, device)

    # Dataloader using ASG's args (to ensure CLIP features are loaded if needed)
    data_root = r"E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUAA-SIRST"
    
    val_loader = make_loader(
        data_root, '50_50/test.txt', size=256, 
        batch_size=1, augment=False, shuffle=False, keep_ratio_pad=False,
        mask_suffix='_pixels0',
        mllm_features_path=os.path.join(data_root, "Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt")
    )

    # We will pick a few specific challenging examples.
    # We can iterate through the dataset until we find highly informative candidates.
    # Good candidates: Targets with significant geometric structure.
    
    out_dir = r"E:\code\EfficientSAM-main\EfficientSAM-main\scripts\visualizations_morphology"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating qualitative heatmap comparisons...")
    
    selected_indices = [15, 60, 120, 203, 311, 400] # Random scattered samples
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i not in selected_indices:
                continue

            path = val_loader.dataset.samples[i][0]
            name = batch["name"][0]
            
            img_bgr = cv2.imread(path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask_label = batch["mask"].squeeze(0).squeeze(0).numpy()
            
            # Predict
            prob_base = inference(model_baseline, args_baseline, batch, device)
            prob_asg = inference(model_asg, args_asg, batch, device)
            
            # Upsample
            prob_base_up = F.interpolate(prob_base.unsqueeze(0).unsqueeze(0), size=(img_gray.shape[0], img_gray.shape[1]), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            prob_asg_up = F.interpolate(prob_asg.unsqueeze(0).unsqueeze(0), size=(img_gray.shape[0], img_gray.shape[1]), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

            # Crop around target for better visualization
            y_i, x_i = np.where(mask_label > 0)
            if len(y_i) == 0:
                continue
            cy, cx = int(np.mean(y_i)), int(np.mean(x_i))
            patch_size = 40
            
            h, w = img_gray.shape
            y1, y2 = max(0, cy - patch_size), min(h, cy + patch_size)
            x1, x2 = max(0, cx - patch_size), min(w, cx + patch_size)
            
            crop_img = img_gray[y1:y2, x1:x2]
            crop_mask = mask_label[y1:y2, x1:x2]
            crop_prob_base = prob_base_up[y1:y2, x1:x2]
            crop_prob_asg = prob_asg_up[y1:y2, x1:x2]
            
            plt.figure(figsize=(16, 4))
            
            plt.subplot(1, 4, 1)
            plt.imshow(crop_img, cmap='gray')
            plt.contour(crop_mask, colors='red', linewidths=1.5, alpha=0.8)
            plt.title('Original Image & GT', fontsize=12)
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(crop_prob_base, cmap='jet', vmin=0, vmax=1)
            plt.contour(crop_mask, colors='white', linewidths=1, alpha=0.5)
            plt.title('Baseline Probability Map\n(Morphological Collapse)', fontsize=12)
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.imshow(crop_prob_asg, cmap='jet', vmin=0, vmax=1)
            plt.contour(crop_mask, colors='white', linewidths=1, alpha=0.5)
            plt.title('ASG Probability Map\n(Structure Preserved)', fontsize=12)
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.imshow(crop_mask, cmap='gray')
            plt.title('Ground Truth Mask', fontsize=12)
            plt.axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{name}_compare.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {out_path}")

    print(f"Done. Visualizations saved into: {out_dir}")

if __name__ == '__main__':
    main()
