import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.measure import label

from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_text_sparse_prompt_projector,
    build_text_dense_mask_prompt_generator_v2,
    build_text_dense_mask_prompt_generator,
    build_gated_backbone_bifusion_block_adapter,
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--data_root', type=str, default=r"E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUDT-SIRST")
parser.add_argument('--split', type=str, default='50_50/test.txt')
parser.add_argument('--mask_suffix', type=str, default='')
cmd_args = parser.parse_args()

class DictArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def calculate_pd_fa(pred, gt):
    pred_labeled, num_pred = label(pred, return_num=True, connectivity=2)
    gt_labeled, num_gt = label(gt, return_num=True, connectivity=2)
    
    tp_objects = 0
    for i in range(1, num_gt + 1):
        if np.any((gt_labeled == i) & pred):
            tp_objects += 1
            
    fp_pixels = np.sum(pred & (~gt))
    total_pixels = pred.size
    
    return tp_objects, num_gt, fp_pixels, total_pixels

def build_model_from_ckpt(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if 'args' not in ckpt:
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

def inference_logits(model, args, batch, device):
    img = batch["image"].to(device)
    mask_label = batch["mask"].squeeze().numpy() 

    clip_feat = batch.get("clip_text_feat", None)
    clip_token_feat = batch.get("clip_text_token_feat", None)
    clip_token_mask = batch.get("clip_text_attn_mask", None)
    
    if clip_feat is not None: clip_feat = clip_feat.to(device)
    if clip_token_feat is not None: clip_token_feat = clip_token_feat.to(device)
    if clip_token_mask is not None: clip_token_mask = clip_token_mask.to(device)
    
    # Import exact prompt strategy from training script
    from train_sirst_hq_ubuntu import sample_points_from_mask
    masks = batch["mask"].to(device)
    
    n_pos = getattr(args, 'n_pos', 1)
    n_neg = getattr(args, 'n_neg', 0)
    boundary_prior = bool(getattr(args, 'boundary_prior_sampling', False))
    boundary_ratio = float(getattr(args, 'boundary_ratio', 0.5))
    
    pts, pts_labels = sample_points_from_mask(
        masks.squeeze(1),
        n_pos=n_pos,
        n_neg=n_neg,
        boundary_prior=boundary_prior,
        boundary_ratio=boundary_ratio,
    )
    pts, pts_labels = pts.to(device), pts_labels.to(device)
    
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
    
    logits = predicted_logits.squeeze() 
    return logits, mask_label

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    model, args = build_model_from_ckpt(cmd_args.ckpt, device)

    data_root = cmd_args.data_root
    
    val_loader = make_loader(
        data_root, cmd_args.split, size=256, 
        batch_size=1, augment=False, shuffle=False, keep_ratio_pad=False,
        mask_suffix=cmd_args.mask_suffix,
        mllm_features_path=os.path.join(data_root, "Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt")
    )

    print("Evaluating accurately on the dataset...")
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    
    # Storage for fixed thresholding (standard way)
    fixed_miou_list = {t: [] for t in thresholds}
    fixed_f1_list = {t: [] for t in thresholds}
    
    fixed_global_inter = {t: 0 for t in thresholds}
    fixed_global_union = {t: 0 for t in thresholds}
    fixed_tp_objects = {t: 0 for t in thresholds}
    fixed_fp_pixels = {t: 0 for t in thresholds}
    
    # Oracle per-image metrics (reproducing the training script's behavior)
    oracle_miou_list = []
    oracle_f1_list = []
    oracle_inter_list = []
    oracle_union_list = []
    oracle_tp_objects = 0
    oracle_fp_pixels = 0
    
    total_gt_objects = 0
    total_pixels_all = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            logits, mask_label = inference_logits(model, args, batch, device)
            prob_map = torch.sigmoid(logits).cpu().numpy()
            gt = mask_label > 0
            
            # --- Per Image Oracle Tracker ---
            best_img_miou = -1.0
            best_img_t = 0.5
            
            for t in thresholds:
                pred = prob_map >= t
                
                # Intersection and Union
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                
                # Image IoU
                iou = inter / union if union > 0 else 1.0
                
                # Update fixed
                fixed_miou_list[t].append(iou)
                
                eps = 1e-6
                precision = inter / (pred.sum() + eps)
                recall = inter / (gt.sum() + eps)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                fixed_f1_list[t].append(f1)
                
                fixed_global_inter[t] += inter
                fixed_global_union[t] += union
                
                tp_o, n_gt, fp_p, t_p = calculate_pd_fa(pred, gt)
                fixed_tp_objects[t] += tp_o
                fixed_fp_pixels[t] += fp_p
                
                if t == thresholds[0]: # Count GT only once per image
                    total_gt_objects += n_gt
                    total_pixels_all += t_p
                    
                # Track oracle
                if iou > best_img_miou:
                    best_img_miou = iou
                    best_img_t = t
            
            # Record oracle image performance
            pred_oracle = prob_map > best_img_t
            inter_o = np.logical_and(pred_oracle, gt).sum()
            union_o = np.logical_or(pred_oracle, gt).sum()
            oracle_inter_list.append(inter_o)
            oracle_union_list.append(union_o)
            
            oracle_miou_list.append(best_img_miou)
            
            precision_o = inter_o / (pred_oracle.sum() + eps)
            recall_o = inter_o / (gt.sum() + eps)
            f1_o = 2 * precision_o * recall_o / (precision_o + recall_o) if (precision_o + recall_o) > 0 else 0.0
            oracle_f1_list.append(f1_o)
            
            tp_o_o, _, fp_p_o, _ = calculate_pd_fa(pred_oracle, gt)
            oracle_tp_objects += tp_o_o
            oracle_fp_pixels += fp_p_o

    # Find the best FIXED threshold (Standard mIoU)
    best_fixed_miou = 0
    best_t = 0.5
    for t in thresholds:
        mean_miou = np.mean(fixed_miou_list[t])
        if mean_miou > best_fixed_miou:
            best_fixed_miou = mean_miou
            best_t = t
            
    # Calculate dataset-wide metrics for best fixed
    fixed_niou = fixed_global_inter[best_t] / fixed_global_union[best_t] if fixed_global_union[best_t] > 0 else 1.0
    fixed_f1_avg = np.mean(fixed_f1_list[best_t])
    fixed_pd = fixed_tp_objects[best_t] / total_gt_objects if total_gt_objects > 0 else 0
    fixed_fa = (fixed_fp_pixels[best_t] / total_pixels_all) * 1e6
    
    # Calculate oracle metrics
    oracle_miou = np.mean(oracle_miou_list)
    # The training script logic calculated nIoU as the mean of per-image IoUs!
    oracle_niou_training = np.mean(oracle_miou_list)
    # True nIoU for oracle outputs
    oracle_niou_true = sum(oracle_inter_list) / sum(oracle_union_list) if sum(oracle_union_list) > 0 else 1.0
    
    oracle_f1_avg = np.mean(oracle_f1_list)
    oracle_pd = oracle_tp_objects / total_gt_objects if total_gt_objects > 0 else 0
    oracle_fa = (oracle_fp_pixels / total_pixels_all) * 1e6

    print("\n============== EVALUATION RESULTS ==============")
    print(f"Dataset:      {data_root}")
    print(f"Images:       {len(oracle_miou_list)}")
    print(f"----------------------------------------------")
    print(f">> STANDARD (Realistic) METRICS at optimal fixed threshold (Thr = {best_t:.2f})")
    print(f"  mIoU (%):     {fixed_niou * 100:.2f}  <-- (Global Intersection / Global Union)")
    print(f"  nIoU (%):     {best_fixed_miou * 100:.2f}  <-- (Mean of per-image IoUs)")
    print(f"  F1-Score (%): {fixed_f1_avg * 100:.2f}")
    print(f"  Pd (%):       {fixed_pd * 100:.2f}")
    print(f"  Fa (1e-6):    {fixed_fa:.2f}")
    print(f"----------------------------------------------")
    print(f">> ORACLE METRICS (Training Script Method - Adaptive Thresholding)")
    print(f"  mIoU (%):     {oracle_niou_true * 100:.2f}  <-- (Global Intersection / Global Union)")
    print(f"  nIoU (%):     {oracle_miou * 100:.2f}  <-- (Mean of per-image IoUs, matches checkpoint name)")
    print(f"  F1-Score (%): {oracle_f1_avg * 100:.2f}")
    print(f"  Pd (%):       {oracle_pd * 100:.2f}")
    print(f"  Fa (1e-6):    {oracle_fa:.2f}")
    print("================================================")

if __name__ == '__main__':
    main()
