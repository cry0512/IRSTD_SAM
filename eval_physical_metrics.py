import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_text_sparse_prompt_projector,
    build_text_dense_mask_prompt_generator_v2,
    build_text_dense_mask_prompt_generator,
    build_gated_backbone_bifusion_block_adapter,
)

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint')
cmd_args = parser.parse_args()

CKPT_PATH = cmd_args.ckpt

def calculate_scr_bsf(img_in, img_out, gt_mask, neighborhood_size=50):
    img_in = img_in.astype(np.float32)
    img_out = img_out.astype(np.float32)
    gt_mask = (gt_mask > 0)
    
    y_idx, x_idx = np.where(gt_mask)
    if len(y_idx) == 0:
        return None, None
    
    cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
    
    kernel = np.ones((5, 5), np.uint8)
    target_area = cv2.dilate(gt_mask.astype(np.uint8), kernel) > 0
    
    H, W = gt_mask.shape
    x1, x2 = max(0, cx - neighborhood_size), min(W, cx + neighborhood_size)
    y1, y2 = max(0, cy - neighborhood_size), min(H, cy + neighborhood_size)
    
    bg_mask = np.zeros_like(gt_mask)
    bg_mask[y1:y2, x1:x2] = True
    bg_mask = bg_mask & (~target_area)
    
    mu_t_in = np.mean(img_in[target_area])
    mu_b_in = np.mean(img_in[bg_mask])
    sigma_b_in = np.std(img_in[bg_mask]) + 1e-6
    scr_in = abs(mu_t_in - mu_b_in) / sigma_b_in
    
    mu_t_out = np.mean(img_out[target_area])
    mu_b_out = np.mean(img_out[bg_mask])
    sigma_b_out = np.std(img_out[bg_mask]) + 1e-6
    scr_out = abs(mu_t_out - mu_b_out) / sigma_b_out
    
    scrg = 20 * np.log10(scr_out / (scr_in + 1e-6) + 1e-6)
    bsf = sigma_b_in / (sigma_b_out + 1e-6)
    
    return scrg, bsf, scr_in, scr_out

class DictArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Checkpoint to get args
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    
    if 'args' in ckpt:
        args_dict = ckpt['args']
    else:
        raise ValueError("Args not found in checkpoint!")
        
    args = DictArgs(**args_dict)
    
    # Override paths for local testing
    args.data_root = r"E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUAA-SIRST"
    if getattr(args, 'use_mllm_prompt', False):
        args.mllm_features_path = os.path.join(args.data_root, "Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt")

    # 1. Build Model
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
            asg_kwargs = {
                "r_bins": getattr(args, 'asg_radial_bins', 64),
                "theta_bins": getattr(args, 'asg_angular_bins', 128),
            }
        else:
            asg_cls = AnisotropicSpectralGating
            asg_kwargs = {
                "num_radial_bins": getattr(args, 'asg_radial_bins', 64),
                "num_angular_bins": getattr(args, 'asg_angular_bins', 128),
            }

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
    elif getattr(args, 'use_radial_gate_hq', False):
        from efficient_sam.freq_modules import RadialFreqGate
        rgate_loc = getattr(args, 'rgate_loc', 'encoder')
        if rgate_loc in ("encoder", "both"):
            dim_enc = model.image_encoder.neck[0].out_channels
            model.image_encoder.radial_gate = RadialFreqGate(
                dim_enc,
                patch_size=getattr(args, 'freq_patch_size_hq', 8),
                num_bins=getattr(args, 'radial_bins_hq', 6),
                channel_shared=getattr(args, 'radial_channel_shared_hq', False),
                edge_boost=getattr(args, 'rgate_edge_boost', 0.5),
                high_freq_threshold=getattr(args, 'rgate_high_freq_thresh', 0.6),
            ).to(device)
            model.image_encoder.rgate_strength = float(getattr(args, 'rgate_strength_enc', 0.5))
        if rgate_loc in ("decoder", "both"):
            c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
            model.mask_decoder.radial_gate = RadialFreqGate(
                c_dec,
                patch_size=getattr(args, 'freq_patch_size_hq', 8),
                num_bins=getattr(args, 'radial_bins_hq', 6),
                channel_shared=getattr(args, 'radial_channel_shared_hq', False),
                edge_boost=getattr(args, 'rgate_edge_boost', 0.5),
                high_freq_threshold=getattr(args, 'rgate_high_freq_thresh', 0.6),
            ).to(device)
            model.mask_decoder.rgate_strength_dec = float(getattr(args, 'rgate_strength_dec', 0.5))

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
                img_dim=img_dim,
                text_dim=args.mllm_text_dim,
                hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                num_heads=max(1, int(args.text_dense_num_heads)),
            ).to(device)
        else:
            dense_generator = build_text_dense_mask_prompt_generator(
                img_dim=img_dim,
                text_dim=args.mllm_text_dim,
                hidden_dim=max(8, int(args.text_dense_hidden_dim)),
            ).to(device)
    else:
        dense_generator = None

    if getattr(args, "use_gated_bifusion_backbone_blocks", False):
        vision_dim = int(model.image_encoder.patch_embed.proj.out_channels)
        num_layers = len(getattr(model.image_encoder, "blocks", []))
        common_kwargs = dict(
            num_layers=max(1, int(num_layers)),
            vision_dim=vision_dim,
            text_dim=args.mllm_text_dim,
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

    # 2. Load Weights
    print(f"Loading weights from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # The saved model might be wrapped in DataParallel/DDP or have prefix
    sd = ckpt['model']
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # 3. Create DataLoader
    val_loader = make_loader(
        args.data_root, args.val_txt,
        size=args.size, 
        batch_size=1, augment=False, shuffle=False, keep_ratio_pad=args.keep_ratio_pad,
        mask_suffix=getattr(args, 'mask_suffix', ""),
        mllm_features_path=getattr(args, 'mllm_features_path', None) if getattr(args, 'use_mllm_prompt', False) else None
    )

    scrg_list = []
    bsf_list = []
    scr_in_list = []
    scr_out_list = []
    
    # helper for text sparse logic
    def _select_text_sparse_prompt_source(
        args,
        raw_clip_feat,
        fused_clip_feat,
        fused_clip_token_feat=None,
        fused_clip_token_mask=None,
    ):
        source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
        if source == "raw_global":
            if raw_clip_feat is not None:
                return raw_clip_feat, None
            if fused_clip_feat is not None:
                return fused_clip_feat, None
            return None, None
        if source == "fused_global":
            if fused_clip_feat is not None:
                return fused_clip_feat, None
            if raw_clip_feat is not None:
                return raw_clip_feat, None
            return None, None
        if source == "fused_tokens":
            if fused_clip_token_feat is not None:
                return fused_clip_token_feat, fused_clip_token_mask
            if fused_clip_feat is not None:
                return fused_clip_feat, None
            if raw_clip_feat is not None:
                return raw_clip_feat, None
            return None, None
        raise ValueError(f"Unsupported text_sparse_prompt_source: {source}")


    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            img = batch["image"].to(device)
            mask_label = batch["mask"].squeeze(0).squeeze(0).numpy() # Shape [H, W]
            
            # The dataset outputs standard augmented tensors.
            path = val_loader.dataset.samples[i][0]
            
            img_bgr = cv2.imread(path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Predict
            clip_feat = batch.get("clip_text_feat", None)
            clip_token_feat = batch.get("clip_text_token_feat", None)
            clip_token_mask = batch.get("clip_text_attn_mask", None)
            
            if clip_feat is not None:
                clip_feat = clip_feat.to(device)
            if clip_token_feat is not None:
                clip_token_feat = clip_token_feat.to(device)
            if clip_token_mask is not None:
                clip_token_mask = clip_token_mask.to(device)
            
            b, c, h, w = img.shape
            y_idx, x_idx = np.where(mask_label > 0)
            if len(y_idx) > 0:
                cx, cy = float(np.mean(x_idx)), float(np.mean(y_idx))
            else:
                cx, cy = w / 2.0, h / 2.0
                
            # For EfficientSAM HQ, expected point shape is [B, Q, N, 2] and labels [B, Q, N] (B=1, Q=1, N=1)
            pts = torch.tensor([[[[cx, cy]]]], dtype=torch.float32, device=device)
            pts_labels = torch.tensor([[[1]]], dtype=torch.float32, device=device)
            
            # Backbone
            if getattr(args, 'use_gated_bifusion_backbone_blocks', False) and clip_feat is not None:
                text_tokens = clip_token_feat if clip_token_feat is not None else clip_feat.unsqueeze(1)
                t_mask = clip_token_mask if clip_token_mask is not None else torch.ones((text_tokens.shape[0], 1), device=device, dtype=torch.long)
                
                out = model.get_image_embeddings_with_text(img, text_tokens, text_attention_mask=t_mask)
                if len(out) == 4:
                    img_emb, interms, text_tokens_out, text_mask_out = out
                else:
                    img_emb, interms = out[0], out[1]
            else:
                img_emb, interms = model.get_image_embeddings(img)
                
            # Sparse prompt
            sparse_p = None
            if model.text_sparse_projector is not None and clip_feat is not None:
                sparse_source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
                sparse_input, sparse_mask = _select_text_sparse_prompt_source(
                    args,
                    raw_clip_feat=clip_feat,
                    fused_clip_feat=clip_feat,
                    fused_clip_token_feat=clip_token_feat,
                    fused_clip_token_mask=clip_token_mask,
                )
                if sparse_input is not None:
                    if sparse_input.dim() == 3:
                        sparse_p = model.text_sparse_projector(sparse_input, attention_mask=sparse_mask)
                    else:
                        sparse_p = model.text_sparse_projector(
                            sparse_input,
                            use_global_prompt_enhance=(sparse_source == "raw_global"),
                        )
            
            # Dense prompt
            dense_p = None
            if model.text_dense_generator is not None and clip_feat is not None:
                target_size = getattr(model.prompt_encoder, "mask_input_size", None)
                if getattr(model.text_dense_generator, "expects_token_level", False):
                    dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
                    if dense_text_input is not None:
                        dense_p = model.text_dense_generator(
                            img_emb,
                            dense_text_input,
                            attention_mask=clip_token_mask if clip_token_feat is not None else None,
                            output_size=tuple(target_size) if target_size is not None else None,
                        )
                else:
                    dense_p = model.text_dense_generator(
                        img_emb, clip_feat, output_size=tuple(target_size) if target_size is not None else None
                    )
                if dense_p is not None:
                    dense_p = dense_p * float(getattr(args, "text_dense_prompt_scale", 1.0))
            
            # Pad or extract proper size points
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
            
            # Resize probability map to match original image size
            prob_map_up = F.interpolate(prob_map.unsqueeze(0).unsqueeze(0), size=(img_gray.shape[0], img_gray.shape[1]), mode='bilinear', align_corners=False).squeeze()
            prob_map_up = prob_map_up.cpu().numpy()
            # Resize mask label to match original image size to avoid array indexing errors
            # (since dataloader processes fixed square sizes but original image may not be square)
            mask_label_resized = cv2.resize(mask_label, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            result = calculate_scr_bsf(img_gray, prob_map_up, mask_label_resized)
            if result[0] is None:
                continue
            
            scrg, bsf, scr_in, scr_out = result
            
            if scrg is not None and bsf is not None:
                # Discard unrealistic extremes
                bsf = min(bsf, 500.0) 
                
                scrg_list.append(scrg)
                bsf_list.append(bsf)
                scr_in_list.append(scr_in)
                scr_out_list.append(scr_out)

    print(f"\n--- Final Results ({len(scrg_list)} valid targets) ---")
    print(f"Mean SCRG (dB): {np.mean(scrg_list):.4f}")
    print(f"Mean BSF:       {np.mean(bsf_list):.4f}")
    print(f"Median SCR_in:  {np.median(scr_in_list):.4f} (min={np.min(scr_in_list):.4f}, max={np.max(scr_in_list):.4f})")
    print(f"Median SCR_out: {np.median(scr_out_list):.4f} (min={np.min(scr_out_list):.4f}, max={np.max(scr_out_list):.4f})")

if __name__ == '__main__':
    main()
