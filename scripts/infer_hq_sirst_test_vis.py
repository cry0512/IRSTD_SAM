import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_backbone_bifusion_block_adapter,
    build_bifusion_adapter_lite,
    build_gated_backbone_bifusion_block_adapter,
    build_text_conditioner,
    build_text_dense_mask_prompt_generator,
    build_text_dense_mask_prompt_generator_v2,
    build_text_sparse_prompt_projector,
)
from sirst_dataset import _find_with_ext, get_img_norm_cfg, normalize_grayscale
from train_sirst_hq import (
    _apply_backbone_bifusion_adapter,
    _apply_bifusion_adapter,
    _build_text_prompt_inputs,
    _merge_dense_mask_prompts,
    autocast_ctx,
    compute_metrics,
    sample_points_from_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HQ-SAM checkpoint on IRSTD-1k test images and save masks plus visualizations."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split_txt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def io_path(path: str) -> str:
    if os.name != "nt":
        return path
    norm = os.path.abspath(path)
    if norm.startswith("\\\\?\\"):
        return norm
    if len(norm) < 248:
        return norm
    if norm.startswith("\\\\"):
        return "\\\\?\\UNC\\" + norm[2:]
    return "\\\\?\\" + norm


def ns_from_dict(data: Dict) -> SimpleNamespace:
    return SimpleNamespace(**data)


def resolve_existing_path(*candidates: Optional[str]) -> Optional[str]:
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if os.path.exists(io_path(path)):
            return path
    return None


def resolve_data_root(cli_root: Optional[str], ckpt_args: SimpleNamespace) -> str:
    dataset_name = os.path.basename(str(getattr(ckpt_args, "data_root", "")).rstrip("\\/")) or "IRSTD-1k"
    path = resolve_existing_path(
        cli_root,
        getattr(ckpt_args, "data_root", None),
        os.path.join("E:\\code\\SIRST-5K-main\\SIRST-5K-main\\dataset", dataset_name),
        os.path.join(os.getcwd(), "IRSAM_datasets", dataset_name),
    )
    if path is None:
        raise FileNotFoundError("Could not resolve dataset root. Pass --data_root explicitly.")
    return path


def resolve_split_txt(cli_split: Optional[str], data_root: str, ckpt_args: SimpleNamespace) -> str:
    split_txt = cli_split or getattr(ckpt_args, "val_txt", "test.txt")
    dataset_name = os.path.basename(data_root.rstrip("\\/"))
    path = resolve_existing_path(
        split_txt,
        os.path.join(data_root, split_txt),
        os.path.join(os.getcwd(), "IRSAM_datasets", dataset_name, os.path.basename(split_txt)),
    )
    if path is None:
        raise FileNotFoundError(f"Could not resolve split txt for: {split_txt}")
    return path


def resolve_mllm_features_path(data_root: str, ckpt_args: SimpleNamespace) -> Optional[str]:
    if not bool(getattr(ckpt_args, "use_mllm_prompt", False)):
        return None
    path = getattr(ckpt_args, "mllm_features_path", None)
    if not path:
        return None
    resolved = resolve_existing_path(
        path,
        os.path.join(data_root, path),
        os.path.join(data_root, os.path.basename(path)),
    )
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve MLLM feature file: {path}")
    return resolved


def apply_mask_suffix(name: str, mask_suffix: str) -> str:
    if not mask_suffix:
        return name
    base, ext = os.path.splitext(name)
    return base + mask_suffix + ext


class MLLMFeatureStore:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.features = None
        self.mode = None
        self.text_dim = 512
        self.token_len = 77
        if path is None:
            return
        self.features = torch.load(io_path(path), map_location="cpu")
        if isinstance(self.features, dict) and self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict) and "token_features" in first_val:
                self.mode = "token_level"
                token_feat = first_val.get("token_features", None)
                global_feat = first_val.get("global_feat", None)
                if torch.is_tensor(token_feat) and token_feat.dim() == 2:
                    self.token_len = int(token_feat.shape[0])
                    self.text_dim = int(token_feat.shape[1])
                if torch.is_tensor(global_feat) and global_feat.numel() > 0:
                    self.text_dim = int(global_feat.numel())
            else:
                self.mode = "global"
                if torch.is_tensor(first_val) and first_val.numel() > 0:
                    self.text_dim = int(first_val.numel())

    def get(self, stem: str) -> Dict[str, torch.Tensor]:
        if self.features is None:
            return {}
        item = self.features.get(stem, None)
        if self.mode == "token_level":
            zero_global = torch.zeros(self.text_dim, dtype=torch.float32)
            zero_tokens = torch.zeros((self.token_len, self.text_dim), dtype=torch.float32)
            zero_mask = torch.zeros((self.token_len,), dtype=torch.long)
            if not isinstance(item, dict):
                return {
                    "clip_text_feat": zero_global,
                    "clip_text_token_feat": zero_tokens,
                    "clip_text_attn_mask": zero_mask,
                }
            global_feat = item.get("global_feat", zero_global)
            token_feat = item.get("token_features", zero_tokens)
            attn_mask = item.get("attention_mask", zero_mask)
            if not torch.is_tensor(global_feat):
                global_feat = zero_global
            global_feat = global_feat.float().view(-1)
            if global_feat.numel() != self.text_dim:
                out = zero_global.clone()
                use_n = min(self.text_dim, int(global_feat.numel()))
                out[:use_n] = global_feat[:use_n]
                global_feat = out

            if not torch.is_tensor(token_feat) or token_feat.dim() != 2:
                token_feat = zero_tokens
            else:
                tf = torch.zeros((self.token_len, self.text_dim), dtype=torch.float32)
                use_l = min(self.token_len, int(token_feat.shape[0]))
                use_d = min(self.text_dim, int(token_feat.shape[1]))
                tf[:use_l, :use_d] = token_feat[:use_l, :use_d].float()
                token_feat = tf

            if not torch.is_tensor(attn_mask):
                attn_mask = zero_mask
            else:
                attn_mask = attn_mask.long().view(-1)
                if attn_mask.numel() != self.token_len:
                    out = zero_mask.clone()
                    use_n = min(self.token_len, int(attn_mask.numel()))
                    out[:use_n] = attn_mask[:use_n]
                    attn_mask = out
            return {
                "clip_text_feat": global_feat,
                "clip_text_token_feat": token_feat,
                "clip_text_attn_mask": attn_mask,
            }

        if torch.is_tensor(item):
            feat = item.float().view(-1)
            if feat.numel() != self.text_dim:
                out = torch.zeros(self.text_dim, dtype=torch.float32)
                use_n = min(self.text_dim, int(feat.numel()))
                out[:use_n] = feat[:use_n]
                feat = out
            return {"clip_text_feat": feat}
        return {"clip_text_feat": torch.zeros(self.text_dim, dtype=torch.float32)}


def build_modules(ckpt_args: SimpleNamespace, device: str):
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=None,
        use_adapter=bool(getattr(ckpt_args, "use_fs_adapter", False)),
        use_ms_fusion=bool(getattr(ckpt_args, "use_ms_fusion", False)),
        use_detail_enhancer=bool(getattr(ckpt_args, "use_detail_enhancer", False)),
        early_exit_layer=int(getattr(ckpt_args, "early_exit_layer", 0)),
    )

    text_conditioner = None
    text_sparse_prompt = None
    text_dense_prompt = None
    bifusion_adapter = None
    backbone_bifusion_adapter = None

    if bool(getattr(ckpt_args, "use_mllm_prompt", False)):
        img_dim = 256
        if not bool(getattr(ckpt_args, "disable_text_conditioner", False)):
            text_conditioner = build_text_conditioner(
                img_dim=img_dim,
                text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
            ).to(device)
        if bool(getattr(ckpt_args, "use_text_sparse_prompt", False)):
            text_sparse_prompt = build_text_sparse_prompt_projector(
                text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
                embed_dim=int(getattr(model.prompt_encoder, "embed_dim", 256)),
                num_tokens=max(1, int(getattr(ckpt_args, "text_sparse_num_tokens", 1))),
                init_scale=float(getattr(ckpt_args, "text_sparse_init_scale", 0.02)),
                use_raw_global_gate=bool(getattr(ckpt_args, "text_sparse_raw_global_gate", False)),
                raw_global_gate_init_bias=float(getattr(ckpt_args, "text_sparse_raw_global_gate_init_bias", -2.0)),
            ).to(device)
        if bool(getattr(ckpt_args, "use_text_dense_prompt", False)):
            dense_variant = str(getattr(ckpt_args, "text_dense_prompt_type", "global"))
            if dense_variant == "token_xattn":
                text_dense_prompt = build_text_dense_mask_prompt_generator_v2(
                    img_dim=img_dim,
                    text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
                    hidden_dim=max(8, int(getattr(ckpt_args, "text_dense_hidden_dim", 128))),
                    num_heads=max(1, int(getattr(ckpt_args, "text_dense_num_heads", 4))),
                ).to(device)
            else:
                text_dense_prompt = build_text_dense_mask_prompt_generator(
                    img_dim=img_dim,
                    text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
                    hidden_dim=max(8, int(getattr(ckpt_args, "text_dense_hidden_dim", 128))),
                ).to(device)
        if bool(getattr(ckpt_args, "use_bifusion_adapter", False)):
            try:
                interms_dim = int(model.image_encoder.patch_embed.proj.out_channels)
            except Exception:
                interms_dim = int(getattr(ckpt_args, "bifusion_interms_dim", 192))
            bifusion_adapter = build_bifusion_adapter_lite(
                img_dim=img_dim,
                interms_dim=interms_dim,
                text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
                hidden_dim=max(8, int(getattr(ckpt_args, "bifusion_hidden_dim", 128))),
                num_heads=max(1, int(getattr(ckpt_args, "bifusion_num_heads", 4))),
                use_interms_level=not bool(getattr(ckpt_args, "bifusion_disable_interms_level", False)),
                img_res_scale=float(getattr(ckpt_args, "bifusion_img_res_scale", 1.0)),
                interms_res_scale=float(getattr(ckpt_args, "bifusion_interms_res_scale", 1.0)),
                text_res_scale=float(getattr(ckpt_args, "bifusion_text_res_scale", 1.0)),
            ).to(device)

        use_plain_backbone_bifusion = bool(getattr(ckpt_args, "use_bifusion_backbone_blocks", False))
        use_gated_backbone_bifusion = bool(getattr(ckpt_args, "use_gated_bifusion_backbone_blocks", False))
        if use_plain_backbone_bifusion and use_gated_backbone_bifusion:
            use_plain_backbone_bifusion = False
        if use_plain_backbone_bifusion or use_gated_backbone_bifusion:
            try:
                vision_dim = int(model.image_encoder.patch_embed.proj.out_channels)
            except Exception:
                vision_dim = int(getattr(ckpt_args, "bifusion_interms_dim", 192))
            num_layers = len(getattr(model.image_encoder, "blocks", []))
            common_kwargs = dict(
                num_layers=max(1, int(num_layers)),
                vision_dim=vision_dim,
                text_dim=int(getattr(ckpt_args, "mllm_text_dim", 512)),
                hidden_dim=max(8, int(getattr(ckpt_args, "bifusion_hidden_dim", 128))),
                num_heads=max(1, int(getattr(ckpt_args, "bifusion_num_heads", 4))),
                apply_every=max(1, int(getattr(ckpt_args, "bifusion_block_apply_every", 1))),
                vision_res_scale=float(getattr(ckpt_args, "bifusion_block_vision_res_scale", 1.0)),
                text_res_scale=float(getattr(ckpt_args, "bifusion_block_text_res_scale", 1.0)),
            )
            if use_gated_backbone_bifusion:
                backbone_bifusion_adapter = build_gated_backbone_bifusion_block_adapter(
                    gate_hidden_dim=int(getattr(ckpt_args, "bifusion_gate_hidden_dim", 0)),
                    gate_init_bias=float(getattr(ckpt_args, "bifusion_gate_init_bias", -2.0)),
                    **common_kwargs,
                ).to(device)
            else:
                backbone_bifusion_adapter = build_backbone_bifusion_block_adapter(
                    **common_kwargs,
                ).to(device)
            if hasattr(model.image_encoder, "set_text_block_fuser"):
                model.image_encoder.set_text_block_fuser(backbone_bifusion_adapter)
            else:
                model.image_encoder.block_text_fuser = backbone_bifusion_adapter

    model.to(device)
    return {
        "model": model,
        "text_conditioner": text_conditioner,
        "text_sparse_prompt": text_sparse_prompt,
        "text_dense_prompt": text_dense_prompt,
        "bifusion_adapter": bifusion_adapter,
        "backbone_bifusion_adapter": backbone_bifusion_adapter,
    }


def load_checkpoint_modules(modules: Dict[str, Optional[torch.nn.Module]], checkpoint: Dict):
    missing = modules["model"].load_state_dict(checkpoint["model"], strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(
            f"Model state_dict mismatch. Missing={missing.missing_keys}, unexpected={missing.unexpected_keys}"
        )
    for key in [
        "text_conditioner",
        "text_sparse_prompt",
        "text_dense_prompt",
        "bifusion_adapter",
        "backbone_bifusion_adapter",
    ]:
        module = modules.get(key, None)
        state = checkpoint.get(key, None)
        if module is None and state is None:
            continue
        if module is None and state is not None:
            raise RuntimeError(f"Checkpoint contains '{key}' but the module was not constructed.")
        if module is not None and state is None:
            raise RuntimeError(f"Module '{key}' was constructed but checkpoint does not contain its weights.")
        strict = True
        if key == "text_sparse_prompt":
            strict = False
        load_result = module.load_state_dict(state, strict=strict)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        if key == "text_sparse_prompt":
            allowed_missing_prefixes = ("raw_global_norm.", "raw_global_gate.")
            missing_keys = [
                name for name in missing_keys
                if not name.startswith(allowed_missing_prefixes)
            ]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"State mismatch for '{key}'. Missing={missing_keys}, unexpected={unexpected_keys}"
            )


def load_image_tensor(image_path: str, dataset_name: str, use_sc_preproc: bool) -> Tuple[torch.Tensor, Image.Image]:
    vis_image = Image.open(image_path).convert("RGB")
    if not use_sc_preproc:
        return TF.to_tensor(vis_image), vis_image

    try:
        gray = Image.open(image_path).convert("I")
    except Exception:
        gray = Image.open(image_path).convert("L")
    img_np = np.array(gray, dtype=np.float32)
    img_np = normalize_grayscale(img_np, get_img_norm_cfg(dataset_name))
    img_t = torch.from_numpy(np.ascontiguousarray(img_np[None, ...]))
    return img_t, vis_image


def load_mask_tensor(mask_path: str) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask, dtype=np.uint8)
    return torch.from_numpy((mask_np > 127).astype(np.float32))


def resize_image_tensor(image_t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if tuple(image_t.shape[-2:]) == (target_h, target_w):
        return image_t
    resized = F.interpolate(
        image_t.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0]


def draw_points(image: Image.Image, points_xy: torch.Tensor, labels: torch.Tensor, radius: int = 5) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for point, label in zip(points_xy.tolist(), labels.tolist()):
        if label < 0:
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        color = (0, 255, 0) if int(label) == 1 else (0, 180, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
    return out


def overlay_mask(base_rgb: Image.Image, pred_mask_hw: np.ndarray) -> Image.Image:
    base = np.asarray(base_rgb, dtype=np.uint8).copy()
    mask = pred_mask_hw.astype(bool)
    if mask.any():
        base[mask, 0] = 255
        base[mask, 1] = (base[mask, 1] * 0.35).astype(np.uint8)
        base[mask, 2] = (base[mask, 2] * 0.35).astype(np.uint8)
    return Image.fromarray(base)


def make_panel(image_rgb: Image.Image, gt_mask_hw: np.ndarray, pred_mask_hw: np.ndarray, overlay_rgb: Image.Image) -> Image.Image:
    w, h = image_rgb.size
    gt_img = Image.fromarray((gt_mask_hw.astype(np.uint8) * 255), mode="L").convert("RGB")
    pred_img = Image.fromarray((pred_mask_hw.astype(np.uint8) * 255), mode="L").convert("RGB")
    panel = Image.new("RGB", (w * 4, h), color=(0, 0, 0))
    panel.paste(image_rgb, (0, 0))
    panel.paste(gt_img, (w, 0))
    panel.paste(pred_img, (w * 2, 0))
    panel.paste(overlay_rgb, (w * 3, 0))
    return panel


def main():
    cli_args = parse_args()
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli_args.seed)

    checkpoint_path = os.path.abspath(cli_args.checkpoint)
    checkpoint = torch.load(io_path(checkpoint_path), map_location="cpu")
    ckpt_args = ns_from_dict(dict(checkpoint["args"]))

    data_root = resolve_data_root(cli_args.data_root, ckpt_args)
    split_txt = resolve_split_txt(cli_args.split_txt, data_root, ckpt_args)
    mllm_features_path = resolve_mllm_features_path(data_root, ckpt_args)
    feature_store = MLLMFeatureStore(mllm_features_path)

    out_dir = cli_args.out_dir
    if not out_dir:
        ckpt_stem = Path(checkpoint_path).stem
        out_dir = os.path.join(os.path.dirname(checkpoint_path), f"infer_test_vis_{ckpt_stem}")
    out_dir = os.path.abspath(out_dir)
    out_dir_io = io_path(out_dir)
    mask_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlays")
    panel_dir = os.path.join(out_dir, "panels")
    mask_dir_io = io_path(mask_dir)
    overlay_dir_io = io_path(overlay_dir)
    panel_dir_io = io_path(panel_dir)
    os.makedirs(mask_dir_io, exist_ok=True)
    os.makedirs(overlay_dir_io, exist_ok=True)
    os.makedirs(panel_dir_io, exist_ok=True)

    modules = build_modules(ckpt_args, cli_args.device)
    load_checkpoint_modules(modules, checkpoint)
    for module in modules.values():
        if module is not None:
            module.eval()

    model = modules["model"]
    text_conditioner = modules["text_conditioner"]
    text_sparse_prompt = modules["text_sparse_prompt"]
    text_dense_prompt = modules["text_dense_prompt"]
    bifusion_adapter = modules["bifusion_adapter"]
    backbone_bifusion_adapter = modules["backbone_bifusion_adapter"]

    with open(split_txt, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
    if cli_args.max_samples > 0:
        names = names[: cli_args.max_samples]

    dataset_name = os.path.basename(data_root.rstrip("\\/"))
    image_root = os.path.join(data_root, "images")
    mask_root = os.path.join(data_root, "masks")
    mask_suffix = str(getattr(ckpt_args, "mask_suffix", "") or "")

    metrics_iou = []
    metrics_f1 = []
    ckpt_epoch = int(checkpoint.get("epoch", 0))

    for idx, name in enumerate(names, start=1):
        image_path = _find_with_ext(image_root, name)
        mask_name = apply_mask_suffix(name, mask_suffix)
        mask_path = _find_with_ext(mask_root, mask_name)
        if mask_path is None and mask_suffix:
            mask_path = _find_with_ext(mask_root, name)
        if image_path is None or mask_path is None:
            raise FileNotFoundError(f"Missing image or mask for sample: {name}")

        image_t, vis_image = load_image_tensor(
            image_path=image_path,
            dataset_name=dataset_name,
            use_sc_preproc=bool(getattr(ckpt_args, "sctransnet_preproc", False)),
        )
        mask_t = load_mask_tensor(mask_path)
        h, w = int(mask_t.shape[0]), int(mask_t.shape[1])
        if tuple(image_t.shape[-2:]) != (h, w):
            image_t = resize_image_tensor(image_t, (h, w))
        if vis_image.size != (w, h):
            vis_image = vis_image.resize((w, h), resample=Image.BILINEAR)

        images = image_t.unsqueeze(0).to(cli_args.device, non_blocking=True)
        masks = mask_t.unsqueeze(0).to(cli_args.device, non_blocking=True)
        pts, lbl = sample_points_from_mask(
            masks,
            n_pos=int(getattr(ckpt_args, "n_pos", 4)),
            n_neg=int(getattr(ckpt_args, "n_neg", 4)),
            boundary_prior=bool(getattr(ckpt_args, "boundary_prior_sampling", False)),
            boundary_ratio=float(getattr(ckpt_args, "boundary_ratio", 0.5)),
        )
        pts = pts.to(cli_args.device, non_blocking=True)
        lbl = lbl.to(cli_args.device, non_blocking=True)

        text_inputs = feature_store.get(Path(image_path).stem)
        clip_feat = None
        raw_clip_feat = None
        clip_token_feat = None
        clip_token_mask = None
        if text_inputs:
            if "clip_text_feat" in text_inputs:
                clip_feat = text_inputs["clip_text_feat"].unsqueeze(0).to(cli_args.device, non_blocking=True)
                raw_clip_feat = clip_feat
            if "clip_text_token_feat" in text_inputs:
                clip_token_feat = text_inputs["clip_text_token_feat"].unsqueeze(0).to(cli_args.device, non_blocking=True)
            if "clip_text_attn_mask" in text_inputs:
                clip_token_mask = text_inputs["clip_text_attn_mask"].unsqueeze(0).to(cli_args.device, non_blocking=True)

        with torch.inference_mode():
            with autocast_ctx(cli_args.device):
                if backbone_bifusion_adapter is not None:
                    img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                        model=model,
                        backbone_bifusion_adapter=backbone_bifusion_adapter,
                        images=images,
                        clip_feat=clip_feat,
                        clip_token_feat=clip_token_feat,
                        clip_token_mask=clip_token_mask,
                    )
                else:
                    img_emb, interms = model.get_image_embeddings(images)

                if bifusion_adapter is not None:
                    img_emb, interms, clip_feat, clip_token_feat, clip_token_mask = _apply_bifusion_adapter(
                        bifusion_adapter=bifusion_adapter,
                        img_emb=img_emb,
                        interms=interms,
                        clip_feat=clip_feat,
                        clip_token_feat=clip_token_feat,
                        clip_token_mask=clip_token_mask,
                    )

                if text_conditioner is not None and clip_feat is not None:
                    img_emb = text_conditioner(img_emb, clip_feat)

                text_sparse_embeds, text_dense_mask = _build_text_prompt_inputs(
                    model,
                    ckpt_args,
                    img_emb,
                    clip_feat,
                    raw_clip_feat=raw_clip_feat,
                    clip_token_feat=clip_token_feat,
                    clip_token_mask=clip_token_mask,
                    text_sparse_prompt=text_sparse_prompt,
                    text_dense_prompt=text_dense_prompt,
                )
                mask_prompt_eff = _merge_dense_mask_prompts(
                    None,
                    text_dense_mask,
                    float(getattr(ckpt_args, "text_dense_prompt_merge_alpha", 0.5)),
                )
                use_hq_only = bool(
                    getattr(ckpt_args, "hq_token_only", False)
                    or (
                        int(getattr(ckpt_args, "hq_warmup_epochs", 0)) > 0
                        and ckpt_epoch <= int(getattr(ckpt_args, "hq_warmup_epochs", 0))
                    )
                )
                pred_masks, _ = model.predict_masks(
                    img_emb,
                    interms,
                    pts,
                    lbl,
                    batched_masks=mask_prompt_eff,
                    text_sparse_embeddings=text_sparse_embeds,
                    multimask_output=False,
                    input_h=h,
                    input_w=w,
                    output_h=h,
                    output_w=w,
                    hq_token_only=use_hq_only,
                )
                logits = pred_masks[:, 0, 0, ...].unsqueeze(1).float()

        sample_iou, sample_f1 = compute_metrics(logits, masks.unsqueeze(1), thr=cli_args.thr)
        metrics_iou.append(sample_iou)
        metrics_f1.append(sample_f1)

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred_mask = prob >= float(cli_args.thr)
        gt_mask = mask_t.numpy() > 0.5

        mask_path_out = os.path.join(mask_dir, f"{Path(image_path).stem}.png")
        Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L").save(io_path(mask_path_out))

        points_vis = pts[0, 0].detach().cpu()
        labels_vis = lbl[0, 0].detach().cpu()
        base_with_points = draw_points(vis_image, points_vis, labels_vis)
        overlay = overlay_mask(base_with_points, pred_mask)
        overlay.save(io_path(os.path.join(overlay_dir, f"{Path(image_path).stem}.png")))
        panel = make_panel(base_with_points, gt_mask, pred_mask, overlay)
        panel.save(io_path(os.path.join(panel_dir, f"{Path(image_path).stem}.png")))

        if idx == 1 or idx % 20 == 0 or idx == len(names):
            print(
                f"[{idx:03d}/{len(names):03d}] {Path(image_path).stem} "
                f"iou={sample_iou:.4f} f1={sample_f1:.4f}"
            )

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "data_root": data_root,
        "split_txt": split_txt,
        "mllm_features_path": mllm_features_path,
        "out_dir": out_dir,
        "prompt_mode": "gt_mask_sampled_points",
        "inference_mode": "full_image",
        "threshold": float(cli_args.thr),
        "seed": int(cli_args.seed),
        "num_samples": len(names),
        "mean_iou": float(np.mean(metrics_iou)) if metrics_iou else 0.0,
        "mean_f1": float(np.mean(metrics_f1)) if metrics_f1 else 0.0,
    }
    with open(io_path(os.path.join(out_dir, "summary.json")), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
