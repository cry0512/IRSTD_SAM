import math
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .freq_modules import FreqGate, RadialFreqGate, SpectralTransformLite, FFTformerDFFNLite


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
    """2D Image to Patch Embedding"""

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
        self.scale = qk_scale or head_dim ** -0.5
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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]) -> torch.Tensor:
    """Resize absolute positional embeddings to (H,W) and drop cls token if present."""
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


class ImageEncoderViT(nn.Module):
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
        use_freq_gate: bool = False,
        use_radial_gate: bool = False,
        freq_patch_size: int = 8,
        radial_bins: int = 6,
        radial_channel_shared: bool = True,
        use_ffc: bool = False,
        use_fftformer: bool = False,
        ffc_fu_kernel: int = 1,
        ffc_use_only_freq: bool = False,
        ffc_fft_norm: str = "ortho",
        fftf_expansion: float = 3.0,
        fftf_patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.image_embedding_size = img_size // (patch_size if patch_size > 0 else 1)
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))

        self.blocks = nn.ModuleList([Block(patch_embed_dim, num_heads, mlp_ratio, True) for _ in range(depth)])

        self.neck = nn.Sequential(
            nn.Conv2d(patch_embed_dim, neck_dims[0], kernel_size=1, bias=False),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(neck_dims[0], neck_dims[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dims[0]),
        )

        self.freq_gate = FreqGate(neck_dims[0], patch_size=freq_patch_size) if use_freq_gate else None
        self.radial_gate = (
            RadialFreqGate(neck_dims[0], patch_size=freq_patch_size, num_bins=radial_bins, channel_shared=radial_channel_shared)
            if use_radial_gate else None
        )
        self.ffc = SpectralTransformLite(neck_dims[0], fu_kernel=ffc_fu_kernel, use_only_freq=ffc_use_only_freq, fft_norm=ffc_fft_norm) if use_ffc else None
        self.fftf = FFTformerDFFNLite(neck_dims[0], expansion=fftf_expansion, patch_size=fftf_patch_size) if use_fftformer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        if self.freq_gate is not None:
            x = x + self.freq_gate(x)
        if self.radial_gate is not None:
            x = x + self.radial_gate(x)
        if self.ffc is not None:
            x = x + self.ffc(x)
        if self.fftf is not None:
            x = x + self.fftf(x)
        return x
