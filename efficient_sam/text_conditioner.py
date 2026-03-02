"""Text conditioning modules for MLLM text-image multimodal training.

Usage:
    # Build conditioner (text_dim = CLIP output dim)
    conditioner = build_text_conditioner(img_dim=256, text_dim=512)

    # Forward: modulate image embedding with pre-computed CLIP text feature
    img_emb = model.get_image_embeddings(images)  # [B, C, h, w]
    text_feat = batch["clip_text_feat"]             # [B, 512]
    img_emb = conditioner(img_emb, text_feat)       # [B, C, h, w]
"""

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast_disabled_ctx(device_type: str):
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:
        if device_type == "cuda" and torch.cuda.is_available():
            try:
                return torch.cuda.amp.autocast(enabled=False)
            except Exception:
                return nullcontext()
        return nullcontext()


class TextConditioner(nn.Module):
    """FiLM (Feature-wise Linear Modulation) conditioner.

    Takes a text feature vector and generates per-channel scale (gamma)
    and shift (beta) to modulate image features.

    img_out = gamma * img_in + beta

    Args:
        img_dim: number of channels of the image feature map.
        text_dim: dimensionality of input text feature (e.g. 512 for CLIP ViT-B/32).
        output_dim: if not None, projects img_dim->output_dim. Defaults to img_dim.
    """

    def __init__(self, img_dim: int = 256, text_dim: int = 512, output_dim: int = None):
        super().__init__()
        out_dim = output_dim or img_dim
        self.gamma_proj = nn.Linear(text_dim, out_dim)
        self.beta_proj = nn.Linear(text_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize gamma to 1 and beta to 0 (identity at start)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, C, H, W] image feature map.
            text_feat: [B, text_dim] text feature vector.

        Returns:
            [B, C, H, W] modulated image feature map.
        """
        text_feat = text_feat.to(self.gamma_proj.weight.dtype)
        gamma = self.gamma_proj(text_feat)  # [B, C]
        beta = self.beta_proj(text_feat)    # [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * img_feat + beta


def build_text_conditioner(img_dim: int = 256, text_dim: int = 512, output_dim: int = None) -> TextConditioner:
    """Factory function for TextConditioner."""
    return TextConditioner(img_dim=img_dim, text_dim=text_dim, output_dim=output_dim)


class TextSparsePromptProjector(nn.Module):
    """Project text features to one or more SAM sparse prompt tokens.

    Supports either:
      - global text vector: [B, D]
      - token sequence: [B, L, D] (+ optional attention mask [B, L])

    The projector keeps a global delta path (for backward compatibility / stable
    initialization) and adds a token-level delta path when sequence features are
    available.
    """

    def __init__(
        self,
        text_dim: int = 512,
        embed_dim: int = 256,
        num_tokens: int = 1,
        init_scale: float = 0.02,
        use_raw_global_gate: bool = False,
        raw_global_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)
        self.use_raw_global_gate = bool(use_raw_global_gate)
        self.base_tokens = nn.Parameter(
            torch.randn(1, self.num_tokens, self.embed_dim) * float(init_scale)
        )
        self.raw_global_norm = nn.LayerNorm(self.text_dim)
        self.delta_proj = nn.Linear(self.text_dim, self.num_tokens * self.embed_dim)
        self.token_delta_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.raw_global_gate = (
            nn.Linear(self.text_dim, self.num_tokens)
            if self.use_raw_global_gate else None
        )
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.token_delta_proj.weight)
        nn.init.zeros_(self.token_delta_proj.bias)
        if self.raw_global_gate is not None:
            nn.init.zeros_(self.raw_global_gate.weight)
            nn.init.constant_(self.raw_global_gate.bias, float(raw_global_gate_init_bias))

    def _masked_mean(self, seq_feat: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return seq_feat.mean(dim=1)
        mask = attention_mask.to(device=seq_feat.device)
        if mask.dim() != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        mask = (mask > 0).to(seq_feat.dtype).unsqueeze(-1)  # [B, L, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (seq_feat * mask).sum(dim=1) / denom

    def _prepare_raw_global(self, text_feat: torch.Tensor) -> torch.Tensor:
        text_feat = F.normalize(text_feat, dim=-1)
        return self.raw_global_norm(text_feat)

    def forward(
        self,
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_global_prompt_enhance: bool = False,
    ) -> torch.Tensor:
        if text_feat.dim() not in (2, 3):
            raise ValueError("text_feat must have shape [B,D] or [B,L,D]")

        text_feat = text_feat.to(self.delta_proj.weight.dtype)

        if text_feat.dim() == 2:
            bsz = text_feat.shape[0]
            proj_input = self._prepare_raw_global(text_feat) if use_global_prompt_enhance else text_feat
            delta = self.delta_proj(proj_input).view(bsz, self.num_tokens, self.embed_dim)
            if use_global_prompt_enhance and self.raw_global_gate is not None:
                gate = torch.sigmoid(self.raw_global_gate(proj_input)).unsqueeze(-1)
                delta = delta * gate
            return self.base_tokens.expand(bsz, -1, -1) + delta

        # Token-level input: [B, L, D]
        bsz, seq_len, _ = text_feat.shape
        pooled = self._masked_mean(text_feat, attention_mask)
        global_delta = self.delta_proj(pooled).view(bsz, self.num_tokens, self.embed_dim)
        out = self.base_tokens.expand(bsz, -1, -1) + global_delta

        if seq_len <= 0 or self.num_tokens <= 0:
            return out

        token_delta = self.token_delta_proj(text_feat)  # [B, L, E]
        k = min(self.num_tokens, seq_len)
        if attention_mask is None:
            out[:, :k, :] = out[:, :k, :] + token_delta[:, :k, :]
            return out

        mask_bool = (attention_mask.to(device=text_feat.device) > 0)
        if mask_bool.shape[:2] != (bsz, seq_len):
            raise ValueError("attention_mask shape must match token sequence [B, L]")
        pos = torch.arange(seq_len, device=text_feat.device).view(1, seq_len).expand(bsz, seq_len)
        masked_pos = pos.masked_fill(~mask_bool, seq_len)
        sel_pos = torch.topk(masked_pos, k=k, dim=1, largest=False).values  # [B, k]
        valid_sel = (sel_pos < seq_len).unsqueeze(-1)
        sel_pos = sel_pos.clamp(max=max(seq_len - 1, 0))
        gathered = token_delta.gather(1, sel_pos.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        out[:, :k, :] = out[:, :k, :] + gathered * valid_sel.to(gathered.dtype)
        return out


class TextDenseMaskPromptGenerator(nn.Module):
    """Generate a dense mask prompt from image embeddings + global text vector.

    Output is a low-amplitude saliency-like map in [0, 1], initialized close to 0
    so enabling the module does not strongly perturb the pretrained prompt path.
    """

    def __init__(
        self,
        img_dim: int = 256,
        text_dim: int = 512,
        hidden_dim: int = 128,
        init_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.img_dim = int(img_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)

        self.img_proj = nn.Sequential(
            nn.Conv2d(self.img_dim, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(1, self.hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        self._init_weights(init_bias=float(init_bias))

    def _init_weights(self, init_bias: float) -> None:
        # Start near zero mask prompt (after sigmoid) to preserve baseline behavior.
        last = self.fuse[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_bias)
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        text_feat = text_feat.to(self.text_proj.weight.dtype)
        x = self.img_proj(img_feat)
        t = self.text_proj(text_feat).unsqueeze(-1).unsqueeze(-1)
        logits = self.fuse(x + t)
        dense = torch.sigmoid(logits)
        if output_size is not None and tuple(dense.shape[-2:]) != tuple(output_size):
            dense = F.interpolate(dense, size=output_size, mode="bilinear", align_corners=False)
        return dense


class TextDenseMaskPromptGeneratorV2(nn.Module):
    """Token-level text-guided dense mask prompt generator (cross-attention).

    Image features are flattened as spatial queries, and text token features are
    used as keys/values. The output is a coarse saliency-like mask prompt.
    """

    expects_token_level = True

    def __init__(
        self,
        img_dim: int = 256,
        text_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        init_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.img_dim = int(img_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0 or (self.hidden_dim % self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.img_q_proj = nn.Conv2d(self.img_dim, self.hidden_dim, kernel_size=1)
        self.text_k_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.text_v_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        self._init_weights(init_bias=float(init_bias))

    def _init_weights(self, init_bias: float) -> None:
        # Start near zero output prompt while allowing gradients to flow.
        nn.init.zeros_(self.text_k_proj.weight)
        nn.init.zeros_(self.text_k_proj.bias)
        nn.init.zeros_(self.text_v_proj.weight)
        nn.init.zeros_(self.text_v_proj.bias)
        last = self.fuse[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_bias)

    def _prepare_text_inputs(
        self,
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Accept either [B, D] or [B, L, D]. [B, D] becomes a single-token sequence.
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    (text_feat.shape[0], 1),
                    device=text_feat.device,
                    dtype=torch.long,
                )
        elif text_feat.dim() != 3:
            raise ValueError("text_feat must have shape [B,D] or [B,L,D]")

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[:2] != text_feat.shape[:2]:
                raise ValueError("attention_mask must have shape [B,L] matching text_feat")
            attention_mask = (attention_mask > 0).to(device=text_feat.device)
            # MultiheadAttention returns NaN if all keys are masked for a sample.
            valid_any = attention_mask.any(dim=1)
            if not bool(valid_any.all()):
                attention_mask = attention_mask.clone()
                text_feat = text_feat.clone()
                bad = ~valid_any
                attention_mask[bad, 0] = True
                text_feat[bad] = 0
        return text_feat, attention_mask

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        text_feat, attention_mask = self._prepare_text_inputs(text_feat, attention_mask)
        device_type = img_feat.device.type
        key_padding_mask = None if attention_mask is None else (~attention_mask)
        with _autocast_disabled_ctx(device_type):
            img_feat_fp32 = img_feat.float()
            text_feat_fp32 = text_feat.float()

            q_map = self.img_q_proj(img_feat_fp32)  # [B, Hc, H, W]
            bsz, hc, h, w = q_map.shape
            q = q_map.flatten(2).transpose(1, 2)  # [B, HW, Hc]
            k = self.text_k_proj(text_feat_fp32)  # [B, L, Hc]
            v = self.text_v_proj(text_feat_fp32)  # [B, L, Hc]

            attn_out, _ = self.cross_attn(
                query=q,
                key=k,
                value=v,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # [B, HW, Hc]
            attn_map = attn_out.transpose(1, 2).reshape(bsz, hc, h, w)

            logits = self.fuse(torch.cat([q_map, attn_map], dim=1))
            dense = torch.sigmoid(logits)
        if output_size is not None and tuple(dense.shape[-2:]) != tuple(output_size):
            dense = F.interpolate(dense, size=output_size, mode="bilinear", align_corners=False)
        return dense


def build_text_sparse_prompt_projector(
    text_dim: int = 512,
    embed_dim: int = 256,
    num_tokens: int = 1,
    init_scale: float = 0.02,
    use_raw_global_gate: bool = False,
    raw_global_gate_init_bias: float = -2.0,
) -> TextSparsePromptProjector:
    return TextSparsePromptProjector(
        text_dim=text_dim,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        init_scale=init_scale,
        use_raw_global_gate=use_raw_global_gate,
        raw_global_gate_init_bias=raw_global_gate_init_bias,
    )


def build_text_dense_mask_prompt_generator(
    img_dim: int = 256,
    text_dim: int = 512,
    hidden_dim: int = 128,
    init_bias: float = -4.0,
) -> TextDenseMaskPromptGenerator:
    return TextDenseMaskPromptGenerator(
        img_dim=img_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        init_bias=init_bias,
    )


def build_text_dense_mask_prompt_generator_v2(
    img_dim: int = 256,
    text_dim: int = 512,
    hidden_dim: int = 128,
    num_heads: int = 4,
    init_bias: float = -4.0,
) -> TextDenseMaskPromptGeneratorV2:
    return TextDenseMaskPromptGeneratorV2(
        img_dim=img_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        init_bias=init_bias,
    )


class BiFusionAdapterLite(nn.Module):
    """Lightweight bidirectional text-vision fusion at two hierarchy levels.

    Levels:
      1) interms (encoder intermediate feature map, typically HWC)
      2) img_emb (neck output, CHW)

    At each level:
      - vision <- text (cross-attention)
      - text <- vision (cross-attention)

    Returns updated image/intermediate features and updated text tokens/global.
    """

    expects_token_level = True

    def __init__(
        self,
        img_dim: int = 256,
        interms_dim: int = 192,
        text_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        use_interms_level: bool = True,
        img_res_scale: float = 1.0,
        interms_res_scale: float = 1.0,
        text_res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_dim = int(img_dim)
        self.interms_dim = int(interms_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.use_interms_level = bool(use_interms_level)
        self.img_res_scale = float(img_res_scale)
        self.interms_res_scale = float(interms_res_scale)
        self.text_res_scale = float(text_res_scale)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0 or (self.hidden_dim % self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # Shared text projection space for both hierarchy levels.
        self.text_in_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.text_out_proj = nn.Linear(self.hidden_dim, self.text_dim)

        # Level-2: img_emb fusion.
        self.img_q_proj = nn.Conv2d(self.img_dim, self.hidden_dim, kernel_size=1)
        self.img_delta_proj = nn.Conv2d(self.hidden_dim, self.img_dim, kernel_size=1)
        self.img_v_from_t = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.img_t_from_v = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

        # Level-1: interms fusion.
        if self.use_interms_level:
            self.inter_q_proj = nn.Conv2d(self.interms_dim, self.hidden_dim, kernel_size=1)
            self.inter_delta_proj = nn.Conv2d(self.hidden_dim, self.interms_dim, kernel_size=1)
            self.inter_v_from_t = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            self.inter_t_from_v = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )

        self._init_weights()

    def _init_weights(self) -> None:
        # Keep behavior close to identity at initialization.
        nn.init.zeros_(self.text_out_proj.weight)
        nn.init.zeros_(self.text_out_proj.bias)
        nn.init.zeros_(self.img_delta_proj.weight)
        nn.init.zeros_(self.img_delta_proj.bias)
        if self.use_interms_level:
            nn.init.zeros_(self.inter_delta_proj.weight)
            nn.init.zeros_(self.inter_delta_proj.bias)

    def _prepare_text_inputs(
        self,
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Accept [B,D] or [B,L,D]
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    (text_feat.shape[0], 1),
                    device=text_feat.device,
                    dtype=torch.long,
                )
        elif text_feat.dim() != 3:
            raise ValueError("text_feat must have shape [B,D] or [B,L,D]")

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[:2] != text_feat.shape[:2]:
                raise ValueError("attention_mask must have shape [B,L] matching text_feat")
            attention_mask = (attention_mask > 0).to(device=text_feat.device)
            # Guard against all-masked samples.
            valid_any = attention_mask.any(dim=1)
            if not bool(valid_any.all()):
                attention_mask = attention_mask.clone()
                text_feat = text_feat.clone()
                bad = ~valid_any
                attention_mask[bad, 0] = True
                text_feat[bad] = 0
        return text_feat, attention_mask

    def _masked_mean(self, seq_feat: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return seq_feat.mean(dim=1)
        mask = (attention_mask > 0).to(seq_feat.dtype).unsqueeze(-1)  # [B,L,1]
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (seq_feat * mask).sum(dim=1) / denom

    def _maybe_hwc_to_chw(self, feat: torch.Tensor, channels: int) -> Tuple[torch.Tensor, bool]:
        if feat.dim() != 4:
            raise ValueError("feature map must have shape [B,C,H,W] or [B,H,W,C]")
        if feat.shape[1] == channels:
            return feat, False
        if feat.shape[-1] == channels:
            return feat.permute(0, 3, 1, 2).contiguous(), True
        raise ValueError(f"feature channels mismatch: expected {channels}, got shape {tuple(feat.shape)}")

    def _vision_text_step(
        self,
        q_map: torch.Tensor,
        text_hidden: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        v_from_t: nn.MultiheadAttention,
        t_from_v: nn.MultiheadAttention,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, hc, h, w = q_map.shape
        q = q_map.flatten(2).transpose(1, 2)  # [B,HW,Hc]
        v_delta, _ = v_from_t(
            query=q,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q_upd = q + v_delta
        t_delta, _ = t_from_v(
            query=text_hidden,
            key=q_upd,
            value=q_upd,
            need_weights=False,
        )
        q_upd_map = q_upd.transpose(1, 2).reshape(bsz, hc, h, w)
        return q_upd_map, t_delta

    def forward(
        self,
        img_feat: torch.Tensor,
        interms_feat: Optional[torch.Tensor],
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        text_feat, attention_mask = self._prepare_text_inputs(text_feat, attention_mask)
        orig_text = text_feat
        text_feat = text_feat.to(self.text_in_proj.weight.dtype)
        text_hidden = self.text_in_proj(text_feat)  # [B,L,Hc]
        key_padding_mask = None if attention_mask is None else (~attention_mask)

        # Level-1: interms
        inter_out = interms_feat
        if self.use_interms_level and interms_feat is not None:
            inter_chw, was_hwc = self._maybe_hwc_to_chw(interms_feat, self.interms_dim)
            q_map = self.inter_q_proj(inter_chw)
            q_upd_map, t_delta = self._vision_text_step(
                q_map=q_map,
                text_hidden=text_hidden,
                key_padding_mask=key_padding_mask,
                v_from_t=self.inter_v_from_t,
                t_from_v=self.inter_t_from_v,
            )
            inter_delta = self.inter_delta_proj(q_upd_map)
            inter_chw = inter_chw + self.interms_res_scale * inter_delta
            text_hidden = text_hidden + self.text_res_scale * t_delta
            inter_out = inter_chw.permute(0, 2, 3, 1).contiguous() if was_hwc else inter_chw

        # Level-2: img_emb
        if img_feat.dim() != 4 or img_feat.shape[1] != self.img_dim:
            raise ValueError(f"img_feat must have shape [B,{self.img_dim},H,W], got {tuple(img_feat.shape)}")
        img_q_map = self.img_q_proj(img_feat)
        img_q_upd_map, t_delta_img = self._vision_text_step(
            q_map=img_q_map,
            text_hidden=text_hidden,
            key_padding_mask=key_padding_mask,
            v_from_t=self.img_v_from_t,
            t_from_v=self.img_t_from_v,
        )
        img_delta = self.img_delta_proj(img_q_upd_map)
        img_out = img_feat + self.img_res_scale * img_delta
        text_hidden = text_hidden + self.text_res_scale * t_delta_img

        # Back to text space; keep padding tokens unchanged.
        text_delta = self.text_out_proj(text_hidden)
        text_out = orig_text + text_delta.to(orig_text.dtype)
        if attention_mask is not None:
            valid = attention_mask.unsqueeze(-1).to(text_out.dtype)
            text_out = text_out * valid + orig_text * (1.0 - valid)
        text_global = self._masked_mean(text_out, attention_mask)

        return img_out, inter_out, text_out, attention_mask, text_global


def build_bifusion_adapter_lite(
    img_dim: int = 256,
    interms_dim: int = 192,
    text_dim: int = 512,
    hidden_dim: int = 128,
    num_heads: int = 4,
    use_interms_level: bool = True,
    img_res_scale: float = 1.0,
    interms_res_scale: float = 1.0,
    text_res_scale: float = 1.0,
) -> BiFusionAdapterLite:
    return BiFusionAdapterLite(
        img_dim=img_dim,
        interms_dim=interms_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_interms_level=use_interms_level,
        img_res_scale=img_res_scale,
        interms_res_scale=interms_res_scale,
        text_res_scale=text_res_scale,
    )


class BackboneBiFusionBlockAdapter(nn.Module):
    """Bidirectional text-vision fusion inside encoder blocks.

    Operates on token sequences:
      - vision tokens: [B, N, C_v]
      - text tokens:   [B, L, C_t] (or [B, C_t], converted to single token)
    """

    expects_token_level = True

    def __init__(
        self,
        num_layers: int,
        vision_dim: int = 192,
        text_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        apply_every: int = 1,
        vision_res_scale: float = 1.0,
        text_res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.vision_dim = int(vision_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.apply_every = max(1, int(apply_every))
        self.vision_res_scale = float(vision_res_scale)
        self.text_res_scale = float(text_res_scale)

        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0 or (self.hidden_dim % self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.vision_in_proj = nn.Linear(self.vision_dim, self.hidden_dim)
        self.vision_out_proj = nn.Linear(self.hidden_dim, self.vision_dim)
        self.text_in_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.text_out_proj = nn.Linear(self.hidden_dim, self.text_dim)

        self.v_from_t = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])
        self.t_from_v = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        # Near-identity initialization.
        nn.init.zeros_(self.vision_out_proj.weight)
        nn.init.zeros_(self.vision_out_proj.bias)
        nn.init.zeros_(self.text_out_proj.weight)
        nn.init.zeros_(self.text_out_proj.bias)

    def prepare_text_inputs(
        self,
        text_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    (text_feat.shape[0], 1),
                    device=text_feat.device,
                    dtype=torch.long,
                )
        elif text_feat.dim() != 3:
            raise ValueError("text_feat must have shape [B,D] or [B,L,D]")

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[:2] != text_feat.shape[:2]:
                raise ValueError("attention_mask must have shape [B,L] matching text_feat")
            attention_mask = (attention_mask > 0).to(device=text_feat.device)
            valid_any = attention_mask.any(dim=1)
            if not bool(valid_any.all()):
                attention_mask = attention_mask.clone()
                text_feat = text_feat.clone()
                bad = ~valid_any
                attention_mask[bad, 0] = True
                text_feat[bad] = 0
        return text_feat, attention_mask

    def _masked_mean(
        self,
        seq_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return seq_feat.mean(dim=1)
        mask = (attention_mask > 0).to(seq_feat.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (seq_feat * mask).sum(dim=1) / denom

    def forward_layer(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if vision_tokens.dim() != 3:
            raise ValueError("vision_tokens must have shape [B,N,C]")
        if text_tokens.dim() != 3:
            raise ValueError("text_tokens must have shape [B,L,C]")
        if (layer_idx + 1) % self.apply_every != 0:
            return vision_tokens, text_tokens, attention_mask

        idx = max(0, min(int(layer_idx), self.num_layers - 1))
        orig_v = vision_tokens
        orig_t = text_tokens

        key_padding_mask = None if attention_mask is None else (~attention_mask)
        with _autocast_disabled_ctx(vision_tokens.device.type):
            v = vision_tokens.float()
            t = text_tokens.float()
            v_h = self.vision_in_proj(v)  # [B,N,H]
            t_h = self.text_in_proj(t)    # [B,L,H]

            v_delta, _ = self.v_from_t[idx](
                query=v_h,
                key=t_h,
                value=t_h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            v_h_upd = v_h + v_delta
            t_delta, _ = self.t_from_v[idx](
                query=t_h,
                key=v_h_upd,
                value=v_h_upd,
                need_weights=False,
            )
            t_h_upd = t_h + t_delta

            v_delta_out = self.vision_out_proj(v_h_upd)
            t_delta_out = self.text_out_proj(t_h_upd)

        v_out = orig_v + self.vision_res_scale * v_delta_out.to(orig_v.dtype)
        t_out = orig_t + self.text_res_scale * t_delta_out.to(orig_t.dtype)
        if attention_mask is not None:
            valid = attention_mask.unsqueeze(-1).to(t_out.dtype)
            t_out = t_out * valid + orig_t * (1.0 - valid)
        return v_out, t_out, attention_mask


def build_backbone_bifusion_block_adapter(
    num_layers: int,
    vision_dim: int = 192,
    text_dim: int = 512,
    hidden_dim: int = 128,
    num_heads: int = 4,
    apply_every: int = 1,
    vision_res_scale: float = 1.0,
    text_res_scale: float = 1.0,
) -> BackboneBiFusionBlockAdapter:
    return BackboneBiFusionBlockAdapter(
        num_layers=num_layers,
        vision_dim=vision_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_every=apply_every,
        vision_res_scale=vision_res_scale,
        text_res_scale=text_res_scale,
    )


class GatedBackboneBiFusionBlockAdapter(BackboneBiFusionBlockAdapter):
    """Backbone BiFusion with lightweight gates on bidirectional updates.

    Compared with the plain backbone BiFusion, this variant keeps the same
    bidirectional cross-attention path but learns a small gate before adding
    text-driven / vision-driven updates. This makes text injection softer and
    more robust when descriptions are noisy.
    """

    def __init__(
        self,
        num_layers: int,
        vision_dim: int = 192,
        text_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        apply_every: int = 1,
        vision_res_scale: float = 1.0,
        text_res_scale: float = 1.0,
        gate_hidden_dim: int = 0,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            apply_every=apply_every,
            vision_res_scale=vision_res_scale,
            text_res_scale=text_res_scale,
        )
        self.gate_hidden_dim = max(16, int(gate_hidden_dim) if int(gate_hidden_dim) > 0 else self.hidden_dim // 4)
        self.gate_init_bias = float(gate_init_bias)

        self.vision_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.gate_hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.text_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.gate_hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self._init_gate_weights()

    def _init_gate_weights(self) -> None:
        for gate_stack in list(self.vision_gates) + list(self.text_gates):
            first = gate_stack[0]
            last = gate_stack[-1]
            if isinstance(first, nn.Linear):
                nn.init.xavier_uniform_(first.weight)
                nn.init.zeros_(first.bias)
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, self.gate_init_bias)

    def forward_layer(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if vision_tokens.dim() != 3:
            raise ValueError("vision_tokens must have shape [B,N,C]")
        if text_tokens.dim() != 3:
            raise ValueError("text_tokens must have shape [B,L,C]")
        if (layer_idx + 1) % self.apply_every != 0:
            return vision_tokens, text_tokens, attention_mask

        idx = max(0, min(int(layer_idx), self.num_layers - 1))
        orig_v = vision_tokens
        orig_t = text_tokens

        key_padding_mask = None if attention_mask is None else (~attention_mask)
        with _autocast_disabled_ctx(vision_tokens.device.type):
            v = vision_tokens.float()
            t = text_tokens.float()
            v_h = self.vision_in_proj(v)  # [B,N,H]
            t_h = self.text_in_proj(t)    # [B,L,H]

            v_delta, _ = self.v_from_t[idx](
                query=v_h,
                key=t_h,
                value=t_h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            v_gate_in = torch.cat(
                [
                    v_h.mean(dim=1),
                    self._masked_mean(t_h, attention_mask),
                    v_delta.mean(dim=1),
                ],
                dim=-1,
            )
            v_gate = torch.sigmoid(self.vision_gates[idx](v_gate_in)).unsqueeze(1)
            v_h_upd = v_h + v_gate * v_delta

            t_delta, _ = self.t_from_v[idx](
                query=t_h,
                key=v_h_upd,
                value=v_h_upd,
                need_weights=False,
            )
            t_gate_in = torch.cat(
                [
                    self._masked_mean(t_h, attention_mask),
                    v_h_upd.mean(dim=1),
                    self._masked_mean(t_delta, attention_mask),
                ],
                dim=-1,
            )
            t_gate = torch.sigmoid(self.text_gates[idx](t_gate_in)).unsqueeze(1)
            t_h_upd = t_h + t_gate * t_delta

            v_delta_out = self.vision_out_proj(v_h_upd)
            t_delta_out = self.text_out_proj(t_h_upd)

        v_out = orig_v + self.vision_res_scale * v_delta_out.to(orig_v.dtype)
        t_out = orig_t + self.text_res_scale * t_delta_out.to(orig_t.dtype)
        if attention_mask is not None:
            valid = attention_mask.unsqueeze(-1).to(t_out.dtype)
            t_out = t_out * valid + orig_t * (1.0 - valid)
        return v_out, t_out, attention_mask


def build_gated_backbone_bifusion_block_adapter(
    num_layers: int,
    vision_dim: int = 192,
    text_dim: int = 512,
    hidden_dim: int = 128,
    num_heads: int = 4,
    apply_every: int = 1,
    vision_res_scale: float = 1.0,
    text_res_scale: float = 1.0,
    gate_hidden_dim: int = 0,
    gate_init_bias: float = -2.0,
) -> GatedBackboneBiFusionBlockAdapter:
    return GatedBackboneBiFusionBlockAdapter(
        num_layers=num_layers,
        vision_dim=vision_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_every=apply_every,
        vision_res_scale=vision_res_scale,
        text_res_scale=text_res_scale,
        gate_hidden_dim=gate_hidden_dim,
        gate_init_bias=gate_init_bias,
    )
