#!/usr/bin/env python
"""Generate text descriptions for IRSTD images using a multimodal LLM (Qwen2.5-Omni / Qwen3-VL)
and pre-compute CLIP text features for training.

Two-stage offline pipeline:
  1. Image → Qwen2.5-Omni → text description (≤30 words)
  2. Text description → CLIP ViT-B/32 → feature vector [512]

Outputs:
  - mllm_descriptions.json  : {filename: description_string}
  - mllm_clip_features.pt   : {filename: tensor[512]}
  - mllm_clip_token_features.pt (optional):
      {filename: {"token_ids": [77], "attention_mask": [77], "token_features": [77,D], "global_feat": [D]}}

Usage (on GPU machine with transformers + Qwen model):
  python scripts/generate_mllm_prompts.py \
    --data_root /path/to/NUDT-SIRST \
    --split_txt "50_50/train.txt" "50_50/test.txt" \
    --model_path /path/to/Qwen2.5-Omni-7B_or_Qwen3-VL-8B-Instruct \
    --clip_model ViT-B/32
"""

import argparse
import gc
import json
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image

# ---- CLIP text encoding ----
try:
    import clip as clip_module
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# ---- Qwen MLLM (Omni / VL) ----
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ==========================================================================
# Prompt template
# ==========================================================================
SYSTEM_PROMPT = (
    "You are an expert in locating small targets from infrared images. "
    "Describe the target's position, shape (irregular, drone, aircraft, ship, or car), "
    "and background details in no more than 30 words."
)


# ==========================================================================
# Image file discovery
# ==========================================================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cuda_cleanup():
    """Best-effort CUDA memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def format_cuda_mem() -> str:
    """Return a short CUDA memory summary for logging."""
    if not torch.cuda.is_available():
        return "cuda: unavailable"
    try:
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        return f"cuda{dev}: alloc={alloc:.0f}MB reserved={reserved:.0f}MB max={max_alloc:.0f}MB"
    except Exception as e:
        return f"cuda mem stats unavailable: {e}"


def parse_torch_dtype(dtype_name: str):
    """Parse torch dtype from CLI string."""
    name = (dtype_name or "float16").lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name == "auto":
        return "auto"
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def build_device_map_and_memory(
    mllm_device_map: str = "auto",
    max_gpu_mem_gb: Optional[float] = None,
    max_cpu_mem_gb: Optional[float] = None,
):
    """Build `device_map` and optional `max_memory` kwargs for HF loading."""
    dm = (mllm_device_map or "auto").lower()
    max_memory = None

    if dm == "auto":
        device_map = "auto"
        if max_gpu_mem_gb is not None or max_cpu_mem_gb is not None:
            max_memory = {}
            if max_gpu_mem_gb is not None:
                max_memory[0] = f"{int(max_gpu_mem_gb)}GiB"
            if max_cpu_mem_gb is not None:
                max_memory["cpu"] = f"{int(max_cpu_mem_gb)}GiB"
    elif dm in ("cuda", "cuda:0", "gpu"):
        device_map = {"": 0}
    elif dm == "cpu":
        device_map = {"": "cpu"}
    else:
        raise ValueError(
            f"Unsupported --mllm_device_map={mllm_device_map}; use auto/cuda/cpu"
        )
    return device_map, max_memory


def find_image(img_dir: str, name: str) -> Optional[str]:
    """Find an image file by stem name, trying common extensions."""
    stem = Path(name).stem
    for ext in IMG_EXTS:
        path = os.path.join(img_dir, stem + ext)
        if os.path.isfile(path):
            return path
    # Try original name as-is
    path = os.path.join(img_dir, name)
    if os.path.isfile(path):
        return path
    return None


def collect_image_names(data_root: str, split_txts: List[str]) -> List[str]:
    """Collect unique image names from one or more split txt files."""
    names = set()
    for txt in split_txts:
        txt_path = txt if os.path.isabs(txt) else os.path.join(data_root, txt)
        if not os.path.isfile(txt_path):
            print(f"[warn] Split file not found: {txt_path}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                n = line.strip()
                if n and not n.startswith("#"):
                    names.add(n)
    return sorted(names)


# ==========================================================================
# Qwen MLLM text generation
# ==========================================================================
def load_qwen_model(
    model_path: str,
    device: str = "cuda",
    mllm_device_map: str = "auto",
    mllm_dtype: str = "float16",
    max_gpu_mem_gb: Optional[float] = None,
    max_cpu_mem_gb: Optional[float] = None,
    quantization: str = "none",
):
    """Load Qwen2.5-Omni / Qwen3-VL / Qwen2.5-VL (or compatible) model and processor."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library is required. Install with: pip install transformers")

    print(f"Loading MLLM from {model_path} ...")
    dtype = parse_torch_dtype(mllm_dtype)
    device_map, max_memory = build_device_map_and_memory(
        mllm_device_map=mllm_device_map,
        max_gpu_mem_gb=max_gpu_mem_gb,
        max_cpu_mem_gb=max_cpu_mem_gb,
    )
    quant_mode = (quantization or "none").lower()
    print(
        f"MLLM load config: device_map={device_map}, dtype={mllm_dtype}, "
        f"quantization={quant_mode}, max_memory={max_memory}"
    )

    common_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        common_kwargs["max_memory"] = max_memory

    if quant_mode != "none":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise ImportError(
                "bitsandbytes is required for --quantization 4bit/8bit. "
                "Install with: pip install bitsandbytes"
            ) from e
        if quant_mode == "8bit":
            common_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_mode == "4bit":
            common_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(torch.bfloat16 if dtype == "auto" else dtype),
            )
        else:
            raise ValueError("Unsupported quantization mode; use none/8bit/4bit")
        # With BnB quantization, keep compute dtype but avoid redundant full-precision weight load config.
        if dtype != "auto":
            common_kwargs["torch_dtype"] = dtype
    else:
        common_kwargs["torch_dtype"] = dtype

    # Try Qwen2.5-Omni first, then Qwen3-VL / Qwen2.5-VL, then generic AutoModel
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            **common_kwargs,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_type = "qwen2.5-omni"
        print(f"Loaded Qwen2.5-Omni model")
    except Exception as e_omni:
        print(f"[warn] Qwen2.5-Omni load failed: {type(e_omni).__name__}: {e_omni}")
        try:
            # Try Qwen3-VL
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **common_kwargs,
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model_type = "qwen3-vl"
            print(f"Loaded Qwen3-VL model")
        except Exception as e_qwen3_vl:
            print(f"[warn] Qwen3-VL load failed: {type(e_qwen3_vl).__name__}: {e_qwen3_vl}")
            try:
                # Try Qwen2.5-VL
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **common_kwargs,
                )
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model_type = "qwen2.5-vl"
                print(f"Loaded Qwen2.5-VL model")
            except Exception as e_vl:
                print(f"[warn] Qwen2.5-VL load failed: {type(e_vl).__name__}: {e_vl}")
                # Generic fallback
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **common_kwargs,
                )
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model_type = "generic"
                print(f"Loaded generic multimodal model")

    model.eval()

    # ---- Free audio-related components (we only need text output) ----
    freed_mb = 0.0
    audio_attrs = ['audio_tower', 'talker']
    containers = [model]
    if hasattr(model, 'thinker'):
        containers.append(model.thinker)
    for container in containers:
        for attr in audio_attrs:
            sub = getattr(container, attr, None)
            if sub is not None:
                n_bytes = sum(p.numel() * p.element_size() for p in sub.parameters())
                freed_mb += n_bytes / (1024 ** 2)
                setattr(container, attr, None)
    if freed_mb > 0:
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Freed ~{freed_mb:.0f} MB by removing audio components')
    print(format_cuda_mem())

    return model, processor, model_type


def generate_description(
    model, processor, model_type: str,
    image_path: str, max_new_tokens: int = 80,
    use_kv_cache: bool = True,
) -> str:
    """Generate text description for an image using the MLLM."""
    with Image.open(image_path) as pil_img:
        image = pil_img.convert("RGB")

    if model_type in ("qwen2.5-omni", "qwen2.5-vl", "qwen3-vl"):
        # Qwen chat format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this infrared image."},
            ]},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input], images=[image],
            return_tensors="pt", padding=True,
        ).to(model.device)
    else:
        # Generic processor
        inputs = processor(
            text=SYSTEM_PROMPT + "\nDescribe this infrared image.",
            images=image,
            return_tensors="pt",
        ).to(model.device)

    image.close()  # free PIL image memory

    with torch.inference_mode():
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=use_kv_cache,
            return_dict_in_generate=False,
        )
        if model_type == "qwen2.5-omni":
            # Omni supports text+audio generation; we explicitly disable audio.
            generate_kwargs["return_audio"] = False
        output = model.generate(**generate_kwargs)
    # Qwen2.5-Omni may return (text_ids, audio); VL/generic models return tensor
    if isinstance(output, tuple):
        output_ids = output[0]  # text token IDs
    else:
        output_ids = output
    # Decode only the generated part
    input_ids_key = "input_ids" if "input_ids" in inputs else "input_token_ids"
    input_len = inputs.get(input_ids_key, torch.tensor([[]])).shape[1]
    generated = output_ids[0, input_len:].detach().cpu()
    text = processor.decode(generated, skip_special_tokens=True).strip()

    # Aggressively free all GPU tensors to prevent VRAM accumulation
    del inputs, output, output_ids, generated
    cuda_cleanup()

    return text


# ==========================================================================
# CLIP text feature extraction
# ==========================================================================
def encode_texts_with_clip(
    descriptions: Dict[str, str],
    clip_model_name: str = "ViT-B/32",
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Encode text descriptions with CLIP text encoder.

    Returns dict mapping filename_stem -> feature tensor [512].
    """
    if not HAS_CLIP:
        raise ImportError("clip library is required. Install with: pip install git+https://github.com/openai/CLIP.git")

    print(f"Loading CLIP {clip_model_name} ...")
    model, _ = clip_module.load(clip_model_name, device=device)
    model.eval()

    stems = list(descriptions.keys())
    texts = [descriptions[s] for s in stems]
    features = {}

    # Batch encode
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_stems = stems[i:i + batch_size]
        tokens = clip_module.tokenize(batch_texts, truncate=True).to(device)
        with torch.inference_mode():
            text_feat = model.encode_text(tokens)  # [B, 512]
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # L2 normalize

        for j, stem in enumerate(batch_stems):
            features[stem] = text_feat[j].cpu()

        del tokens, text_feat
        if isinstance(device, str) and device.startswith("cuda"):
            cuda_cleanup()

    print(f"Encoded {len(features)} descriptions with CLIP {clip_model_name}")
    del model
    if isinstance(device, str) and device.startswith("cuda"):
        cuda_cleanup()
    return features


def encode_texts_with_clip_token_features(
    descriptions: Dict[str, str],
    clip_model_name: str = "ViT-B/32",
    device: str = "cpu",
    batch_size: int = 64,
    store_dtype: str = "float16",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Encode text descriptions with CLIP and keep token-level hidden states.

    Returns dict mapping filename_stem -> {
        "token_ids": LongTensor [L],
        "attention_mask": LongTensor [L],  # 1 valid token, 0 padding
        "token_features": Tensor [L, D],   # token hidden states after ln_final
        "global_feat": Tensor [D],         # pooled/projected and L2-normalized
    }
    """
    if not HAS_CLIP:
        raise ImportError("clip library is required. Install with: pip install git+https://github.com/openai/CLIP.git")

    dtype_name = (store_dtype or "float16").lower()
    if dtype_name in ("fp16", "float16", "half"):
        save_dtype = torch.float16
    elif dtype_name in ("bf16", "bfloat16"):
        save_dtype = torch.bfloat16
    elif dtype_name in ("fp32", "float32"):
        save_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported --clip_token_store_dtype={store_dtype}")

    print(f"Loading CLIP {clip_model_name} for token-level text features ...")
    model, _ = clip_module.load(clip_model_name, device=device)
    model.eval()

    stems = list(descriptions.keys())
    texts = [descriptions[s] for s in stems]
    token_feature_store: Dict[str, Dict[str, torch.Tensor]] = {}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_stems = stems[i:i + batch_size]
        tokens = clip_module.tokenize(batch_texts, truncate=True).to(device)  # [B, 77]

        with torch.inference_mode():
            # OpenAI CLIP text tower internals (token-level hidden states).
            x = model.token_embedding(tokens).type(model.dtype)  # [B, L, width]
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)  # [L, B, width]
            x = model.transformer(x)
            x = x.permute(1, 0, 2)  # [B, L, width]
            x = model.ln_final(x).type(model.dtype)  # [B, L, width]

            # EOT token is the max token id in each sequence for OpenAI CLIP tokenizer.
            eot_idx = tokens.argmax(dim=-1)
            global_feat = x[torch.arange(x.shape[0], device=tokens.device), eot_idx]
            if getattr(model, "text_projection", None) is not None:
                global_feat = global_feat @ model.text_projection
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            attention_mask = (tokens != 0).long()

        for j, stem in enumerate(batch_stems):
            token_feature_store[stem] = {
                "token_ids": tokens[j].detach().cpu().long(),
                "attention_mask": attention_mask[j].detach().cpu().long(),
                "token_features": x[j].detach().cpu().to(save_dtype),
                "global_feat": global_feat[j].detach().cpu().to(save_dtype),
            }

        del tokens, x, global_feat, attention_mask
        if isinstance(device, str) and device.startswith("cuda"):
            cuda_cleanup()

    print(f"Encoded {len(token_feature_store)} descriptions with CLIP token-level features")
    del model
    if isinstance(device, str) and device.startswith("cuda"):
        cuda_cleanup()
    return token_feature_store


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate MLLM text descriptions and CLIP features for IRSTD images")
    parser.add_argument("--data_root", required=True, help="Dataset root directory")
    parser.add_argument("--split_txt", nargs="+", default=["50_50/train.txt", "50_50/test.txt"],
                        help="Split txt files (relative to data_root)")
    parser.add_argument("--img_dir", default="images", help="Image subdirectory name")
    parser.add_argument("--model_path", default=None,
                        help="Path to Qwen2.5-Omni / Qwen3-VL / compatible MLLM model (required unless --skip_mllm)")
    parser.add_argument("--clip_model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--clip_device", default="cpu", help="Device for CLIP encoding")
    parser.add_argument("--clip_batch_size", type=int, default=64, help="CLIP text batch size")
    parser.add_argument("--mllm_device_map", default="auto",
                        help="HF device_map for MLLM loading: auto/cuda/cpu (auto may offload to system RAM)")
    parser.add_argument("--mllm_dtype", default="float16",
                        help="MLLM weight dtype: float16/bfloat16/float32/auto")
    parser.add_argument("--max_gpu_mem_gb", type=float, default=None,
                        help="When mllm_device_map=auto, cap GPU memory used for model loading (GiB)")
    parser.add_argument("--max_cpu_mem_gb", type=float, default=None,
                        help="When mllm_device_map=auto, cap CPU RAM used for model offload (GiB)")
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"],
                        help="Quantize MLLM weights with bitsandbytes to reduce memory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: data_root)")
    parser.add_argument("--descriptions_json", default=None,
                        help="Path to existing descriptions json (default: <output_dir>/mllm_descriptions.json)")
    parser.add_argument("--max_new_tokens", type=int, default=80,
                        help="Max tokens to generate per image")
    parser.add_argument("--disable_kv_cache", action="store_true",
                        help="Disable generation KV cache to reduce peak VRAM (slower but safer)")
    parser.add_argument("--reload_model_every", type=int, default=0,
                        help="Reload MLLM every N images to mitigate VRAM fragmentation/leaks (0=disable)")
    parser.add_argument("--log_cuda_mem", action="store_true",
                        help="Print CUDA memory usage after each image")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing mllm_descriptions.json (skip already described images)")
    parser.add_argument("--skip_mllm", action="store_true",
                        help="Skip Stage 1 Qwen generation and read existing mllm_descriptions.json")
    parser.add_argument("--skip_clip", action="store_true",
                        help="Skip CLIP encoding (only generate descriptions)")
    parser.add_argument("--save_clip_token_features", action="store_true",
                        help="Also save token-level CLIP text features to mllm_clip_token_features.pt")
    parser.add_argument("--clip_token_store_dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Storage dtype for token_features/global_feat in mllm_clip_token_features.pt")
    args = parser.parse_args()

    data_root = args.data_root
    img_dir = os.path.join(data_root, args.img_dir)
    output_dir = args.output_dir or data_root

    os.makedirs(output_dir, exist_ok=True)
    if args.descriptions_json:
        desc_path = args.descriptions_json
        if not os.path.isabs(desc_path):
            desc_path = os.path.join(output_dir, desc_path)
    else:
        desc_path = os.path.join(output_dir, "mllm_descriptions.json")
    feat_path = os.path.join(output_dir, "mllm_clip_features.pt")
    token_feat_path = os.path.join(output_dir, "mllm_clip_token_features.pt")

    # Collect image names
    names = collect_image_names(data_root, args.split_txt)
    print(f"Found {len(names)} images to process")

    # Load existing descriptions if resuming
    descriptions = {}
    if (args.resume or args.skip_mllm) and os.path.isfile(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"Loaded existing descriptions: {len(descriptions)} entries from {desc_path}")
    elif args.skip_mllm and not os.path.isfile(desc_path):
        raise FileNotFoundError(
            f"--skip_mllm was set, but descriptions file was not found: {desc_path}"
        )

    # ---- Stage 1: Generate descriptions with MLLM ----
    remaining = [n for n in names if Path(n).stem not in descriptions]
    if remaining and not args.skip_mllm:
        print(f"\n{'='*60}")
        print(f"Stage 1: Generating descriptions for {len(remaining)} images")
        print(f"{'='*60}")
        if not args.model_path:
            raise ValueError("--model_path is required to run Stage 1 MLLM generation")
        model, processor, model_type = load_qwen_model(
            args.model_path,
            mllm_device_map=args.mllm_device_map,
            mllm_dtype=args.mllm_dtype,
            max_gpu_mem_gb=args.max_gpu_mem_gb,
            max_cpu_mem_gb=args.max_cpu_mem_gb,
            quantization=args.quantization,
        )
        since_reload = 0

        for i, name in enumerate(remaining):
            if args.reload_model_every > 0 and since_reload >= args.reload_model_every:
                print(f"  ... reloading MLLM after {since_reload} images")
                del model, processor
                cuda_cleanup()
                model, processor, model_type = load_qwen_model(
                    args.model_path,
                    mllm_device_map=args.mllm_device_map,
                    mllm_dtype=args.mllm_dtype,
                    max_gpu_mem_gb=args.max_gpu_mem_gb,
                    max_cpu_mem_gb=args.max_cpu_mem_gb,
                    quantization=args.quantization,
                )
                since_reload = 0

            stem = Path(name).stem
            img_path = find_image(img_dir, name)
            if img_path is None:
                print(f"  [{i+1}/{len(remaining)}] SKIP (not found): {name}")
                continue

            try:
                desc = generate_description(
                    model, processor, model_type, img_path, args.max_new_tokens,
                    use_kv_cache=(not args.disable_kv_cache),
                )
                descriptions[stem] = desc
                print(f"  [{i+1}/{len(remaining)}] {stem}: {desc}")
            except Exception as e:
                print(f"  [{i+1}/{len(remaining)}] ERROR {stem}: {e}")
                descriptions[stem] = "An infrared image with unclear features."
                if "out of memory" in str(e).lower():
                    print("  ... detected CUDA OOM, cleaning cache")
                    cuda_cleanup()

            # Free GPU memory after each inference to prevent KV cache accumulation
            cuda_cleanup()
            since_reload += 1
            if args.log_cuda_mem:
                print(f"    {format_cuda_mem()}")

            # Periodically save
            if (i + 1) % 50 == 0:
                with open(desc_path, "w", encoding="utf-8") as f:
                    json.dump(descriptions, f, ensure_ascii=False, indent=2)
                print(f"  ... saved {len(descriptions)} descriptions")

        # Final save
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(descriptions)} descriptions to {desc_path}")

        # Free MLLM memory
        del model, processor
        cuda_cleanup()
    elif args.skip_mllm:
        if remaining:
            print(f"Skipping Stage 1 (--skip_mllm). {len(remaining)} images are missing descriptions.")
        else:
            print("Skipping Stage 1 (--skip_mllm). All descriptions already exist.")
    else:
        print("All descriptions already exist, skipping Stage 1")

    need_global_clip = not args.skip_clip
    need_token_clip = bool(args.save_clip_token_features)

    # ---- Stage 2: Encode with CLIP ----
    if need_global_clip or need_token_clip:
        print(f"\n{'='*60}")
        print(f"Stage 2: Encoding {len(descriptions)} descriptions with CLIP")
        print(f"{'='*60}")
        if need_global_clip:
            features = encode_texts_with_clip(
                descriptions,
                clip_model_name=args.clip_model,
                device=args.clip_device,
                batch_size=args.clip_batch_size,
            )
            torch.save(features, feat_path)
            print(f"Saved CLIP features to {feat_path}")

            # Print stats
            if features:
                sample_feat = next(iter(features.values()))
                print(f"Feature dim: {sample_feat.shape[0]}, dtype: {sample_feat.dtype}")

        if need_token_clip:
            print(f"\n{'='*60}")
            print(f"Stage 2b: Encoding token-level CLIP features for {len(descriptions)} descriptions")
            print(f"{'='*60}")
            token_features = encode_texts_with_clip_token_features(
                descriptions,
                clip_model_name=args.clip_model,
                device=args.clip_device,
                batch_size=args.clip_batch_size,
                store_dtype=args.clip_token_store_dtype,
            )
            torch.save(token_features, token_feat_path)
            print(f"Saved CLIP token-level features to {token_feat_path}")
            if token_features:
                sample = next(iter(token_features.values()))
                tf = sample["token_features"]
                print(
                    "Token feature sample: "
                    f"token_ids={tuple(sample['token_ids'].shape)}, "
                    f"attention_mask={tuple(sample['attention_mask'].shape)}, "
                    f"token_features={tuple(tf.shape)} dtype={tf.dtype}, "
                    f"global_feat={tuple(sample['global_feat'].shape)}"
                )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done! Files saved in {output_dir}:")
    print(f"  - mllm_descriptions.json ({len(descriptions)} entries)")
    if need_global_clip:
        print(f"  - mllm_clip_features.pt")
    if need_token_clip:
        print(f"  - mllm_clip_token_features.pt")
    print(f"{'='*60}")

    # Show sample descriptions
    print("\nSample descriptions:")
    for stem in list(descriptions.keys())[:5]:
        print(f"  {stem}: {descriptions[stem]}")


if __name__ == "__main__":
    main()
