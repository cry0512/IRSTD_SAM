import math
from typing import List, Type, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FSAdapter(nn.Module):
    def __init__(self, dim: int, scale: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.complex_weight = nn.Parameter(torch.randn(1, dim, 1, 1, 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = H
        assert H * W == N, "Input features must be square"
        x_img = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_spatial = self.spatial_conv(x_img)
        x_freq = torch.fft.rfft2(x_img, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight
        x_spectral = torch.fft.irfft2(x_freq, s=(H, W), norm="ortho")
        out = x_spatial + x_spectral
        out = out.reshape(B, C, N).permute(0, 2, 1)
        return self.scale * out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, act_layer=nn.GELU, use_adapter: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.use_adapter = bool(use_adapter)
        if self.use_adapter:
            self.adapter = FSAdapter(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x_norm = self.norm2(x)
        res = self.mlp(x_norm)
        if self.use_adapter:
            res = res + self.adapter(x_norm)
        x = x + res
        return x


@torch.jit.export
def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]) -> torch.Tensor:
    h, w = hw[0], hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class ImageEncoderViTHQ(nn.Module):
    """EfficientSAM encoder variant that also returns early feature for HQ head.

    Returns tuple: (neck_out: [B,C,H,W], interm: [B,H',W',C]).
    When return_multi_scale is enabled, also returns a list of intermediate features.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
        use_adapter: bool = True,
        return_multi_scale: bool = False,
        early_exit_layer: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.image_embedding_size = img_size // (patch_size if patch_size > 0 else 1)
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.return_multi_scale = bool(return_multi_scale)
        self.ms_out_indices = [2, 5, 8, 11]
        self.early_exit_layer = int(early_exit_layer) if early_exit_layer is not None else None
        if self.early_exit_layer is not None and self.early_exit_layer <= 0:
            self.early_exit_layer = None

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))

        self.blocks = nn.ModuleList([
            Block(patch_embed_dim, num_heads, mlp_ratio, True, use_adapter=use_adapter) for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(patch_embed_dim, neck_dims[0], kernel_size=1, bias=False),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(neck_dims[0], neck_dims[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dims[0]),
        )
        # Optional radial frequency gate (to be attached from trainer)
        self.radial_gate = None
        self.rgate_strength = 0.5
        # Optional AFD gate (to be attached from trainer)
        self.afd_gate = None
        self.afd_strength = 0.5
        # Optional MSFE gate (to be attached from trainer)
        self.msfe_gate = None
        self.msfe_strength = 0.5
        # Optional text fusion module to be attached from trainer.
        self.block_text_fuser = None

    def set_text_block_fuser(self, module: Optional[nn.Module]) -> None:
        self.block_text_fuser = module

    def forward(self, x: torch.Tensor):
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C tokens
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x_tok = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])

        interm = None
        multi_scale_feats = []
        max_blocks = len(self.blocks)
        if self.early_exit_layer is not None:
            max_blocks = max(1, min(max_blocks, self.early_exit_layer))
        for i, blk in enumerate(self.blocks):
            x_tok = blk(x_tok)
            if i == 0:  # take very early feature for HQ branch
                x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
                interm = x_grid  # [B, H', W', C]
            if self.return_multi_scale and i in self.ms_out_indices:
                x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
                multi_scale_feats.append(x_grid.permute(0, 3, 1, 2))
            if (i + 1) >= max_blocks:
                break

        x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
        neck_in = x_grid.permute(0, 3, 1, 2)
        neck_out = self.neck(neck_in)
        # apply optional frequency gate on neck output
        if getattr(self, "radial_gate", None) is not None:
            try:
                neck_out = neck_out + float(getattr(self, "rgate_strength", 0.5)) * self.radial_gate(neck_out)
            except Exception:
                # fall back silently if shapes mismatch
                pass
        # apply optional AFD gate on neck output
        if getattr(self, "afd_gate", None) is not None:
            try:
                afd_delta = self.afd_gate(neck_out) - neck_out
                neck_out = neck_out + float(getattr(self, "afd_strength", 0.5)) * afd_delta
            except Exception:
                pass
        # apply optional MSFE gate on neck output
        if getattr(self, "msfe_gate", None) is not None:
            try:
                msfe_delta = self.msfe_gate(neck_out) - neck_out
                neck_out = neck_out + float(getattr(self, "msfe_strength", 0.5)) * msfe_delta
            except Exception:
                pass
        if self.return_multi_scale:
            return neck_out, interm, multi_scale_feats
        return neck_out, interm

    def forward_with_text(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C tokens
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x_tok = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])

        fuser = getattr(self, "block_text_fuser", None)
        if fuser is not None and text_tokens is not None and hasattr(fuser, "prepare_text_inputs"):
            text_tokens, text_attention_mask = fuser.prepare_text_inputs(text_tokens, text_attention_mask)

        interm = None
        multi_scale_feats = []
        max_blocks = len(self.blocks)
        if self.early_exit_layer is not None:
            max_blocks = max(1, min(max_blocks, self.early_exit_layer))
        for i, blk in enumerate(self.blocks):
            x_tok = blk(x_tok)
            if fuser is not None and text_tokens is not None and hasattr(fuser, "forward_layer"):
                x_tok, text_tokens, text_attention_mask = fuser.forward_layer(
                    x_tok,
                    text_tokens,
                    attention_mask=text_attention_mask,
                    layer_idx=i,
                )
            if i == 0:  # take very early feature for HQ branch
                x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
                interm = x_grid  # [B, H', W', C]
            if self.return_multi_scale and i in self.ms_out_indices:
                x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
                multi_scale_feats.append(x_grid.permute(0, 3, 1, 2))
            if (i + 1) >= max_blocks:
                break

        x_grid = x_tok.reshape(x.shape[0], num_patches, num_patches, x.shape[3])
        neck_in = x_grid.permute(0, 3, 1, 2)
        neck_out = self.neck(neck_in)
        # apply optional frequency gate on neck output
        if getattr(self, "radial_gate", None) is not None:
            try:
                neck_out = neck_out + float(getattr(self, "rgate_strength", 0.5)) * self.radial_gate(neck_out)
            except Exception:
                pass
        # apply optional AFD gate on neck output
        if getattr(self, "afd_gate", None) is not None:
            try:
                afd_delta = self.afd_gate(neck_out) - neck_out
                neck_out = neck_out + float(getattr(self, "afd_strength", 0.5)) * afd_delta
            except Exception:
                pass
        # apply optional MSFE gate on neck output
        if getattr(self, "msfe_gate", None) is not None:
            try:
                msfe_delta = self.msfe_gate(neck_out) - neck_out
                neck_out = neck_out + float(getattr(self, "msfe_strength", 0.5)) * msfe_delta
            except Exception:
                pass
        if self.return_multi_scale:
            return neck_out, interm, multi_scale_feats, text_tokens, text_attention_mask
        return neck_out, interm, text_tokens, text_attention_mask
