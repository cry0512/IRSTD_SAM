import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqGate(nn.Module):
    """Lightweight local frequency gate (EDFFN-style) over 8x8 patches.

    - Learns a per-channel, per-frequency real-valued mask in rFFT domain.
    - Applies reflect padding to support arbitrary H, W.
    - Residual-friendly: usually used as `x = x + gate(x)`.
    """

    def __init__(self, dim: int, patch_size: int = 8):
        super().__init__()
        assert patch_size > 0 and patch_size % 2 == 0, "patch_size must be even and > 0"
        self.dim = dim
        self.patch_size = patch_size
        self.register_parameter(
            "fft",
            nn.Parameter(torch.ones(dim, 1, 1, patch_size, patch_size // 2 + 1)),
        )

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        B, C, H, W = x.shape
        hp = (multiple - H % multiple) % multiple
        wp = (multiple - W % multiple) % multiple
        if hp or wp:
            x = F.pad(x, (0, wp, 0, hp), mode="reflect")
        return x, H, W, hp, wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size
        x_pad, H0, W0, hp, wp = self._pad_to_multiple(x, P)
        B, C, H, W = x_pad.shape

        h_tiles = H // P
        w_tiles = W // P
        # [B, C, H, W] -> [B, C, h_tiles, w_tiles, P, P]
        x_patch = (
            x_pad.view(B, C, h_tiles, P, w_tiles, P).permute(0, 1, 2, 4, 3, 5).contiguous()
        )
        x_fft = torch.fft.rfft2(x_patch.float())  # (..., P, P//2+1), complex
        x_fft = x_fft * self.fft  # per-channel, per-frequency real mask
        x_patch = torch.fft.irfft2(x_fft, s=(P, P))
        # [B, C, h_tiles, w_tiles, P, P] -> [B, C, H, W]
        x_rec = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        if hp or wp:
            x_rec = x_rec[:, :, :H0, :W0]
        return x_rec


class RadialFreqGate(nn.Module):
    """Radial-parameterized frequency gate over 8x8 patches.

    - Parameterizes gain by radial frequency r in [0, 1].
    - Supports optional edge-aware boosting for high-frequency components.
    - Residual-friendly: usually used as `x = x + gate(x)`.
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 8,
        num_bins: int = 6,
        channel_shared: bool = True,
        edge_boost: float = 0.0,
        high_freq_threshold: float = 0.6,
    ):
        super().__init__()
        assert patch_size > 0 and patch_size % 2 == 0, "patch_size must be even and > 0"
        assert num_bins >= 2, "num_bins must be >= 2"
        self.dim = dim
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.channel_shared = channel_shared
        self.edge_boost = float(edge_boost)
        self.use_edge_enhance = self.edge_boost > 0.0
        self.high_freq_threshold = float(high_freq_threshold)

        shape = (1 if channel_shared else dim, num_bins)
        self.gain = nn.Parameter(torch.ones(*shape))

        P = patch_size
        ky = torch.arange(P).float()
        ky = torch.minimum(ky, (P - ky))
        kx = torch.arange(P // 2 + 1).float()
        ky2 = ky[:, None] ** 2
        kx2 = kx[None, :] ** 2
        r = torch.sqrt(ky2 + kx2)
        r_max = math.sqrt((P // 2) ** 2 + (P // 2) ** 2) + 1e-8
        r = (r / r_max).clamp(0.0, 1.0)
        self.register_buffer("r_grid", r)

        centers = torch.linspace(0.0, 1.0, steps=num_bins)
        self.register_buffer("centers", centers)
        self.delta = 1.0 / (num_bins - 1)

        if self.use_edge_enhance:
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
            self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
            hf_mask = (r >= self.high_freq_threshold).float()
            self.register_buffer("high_freq_mask", hf_mask)
        else:
            self.register_buffer("sobel_x", torch.zeros(1, 1, 3, 3))
            self.register_buffer("sobel_y", torch.zeros(1, 1, 3, 3))
            self.register_buffer("high_freq_mask", torch.zeros_like(r))

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        B, C, H, W = x.shape
        hp = (multiple - H % multiple) % multiple
        wp = (multiple - W % multiple) % multiple
        if hp or wp:
            x = F.pad(x, (0, wp, 0, hp), mode="reflect")
        return x, H, W, hp, wp

    def _radial_mask(self, C: int, device, dtype) -> torch.Tensor:
        r = self.r_grid.to(device=device, dtype=self.gain.dtype)
        centers = self.centers.to(device=device, dtype=self.gain.dtype)
        delta = self.delta
        w = torch.relu(1.0 - (r.unsqueeze(0) - centers.view(-1, 1, 1)).abs() / delta)
        w = w / (w.sum(dim=0, keepdim=True) + 1e-8)

        if self.channel_shared:
            g = self.gain.view(-1)
            mask = (w * g.view(-1, 1, 1)).sum(dim=0)
            mask = mask.clamp_min(0.0)
            return mask.view(1, 1, 1, *mask.shape).expand(C, 1, 1, *mask.shape)
        else:
            g = self.gain
            wk = w.unsqueeze(0).expand(g.shape[0], -1, -1, -1)
            mask = (wk * g.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).clamp_min(0.0)
            return mask.view(C, 1, 1, *mask.shape[1:])

    def _edge_strength_per_patch(
        self,
        x_mean: torch.Tensor,
        h_tiles: int,
        w_tiles: int,
        pad_hw: tuple[int, int],
    ) -> torch.Tensor:
        hp, wp = pad_hw
        grad_x = F.conv2d(x_mean, self.sobel_x.to(dtype=x_mean.dtype), padding=1)
        grad_y = F.conv2d(x_mean, self.sobel_y.to(dtype=x_mean.dtype), padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        if hp or wp:
            edge = F.pad(edge, (0, wp, 0, hp), mode="reflect")
        P = self.patch_size
        edge_patch = edge.view(edge.shape[0], 1, h_tiles, P, w_tiles, P).permute(0, 1, 2, 4, 3, 5)
        edge_strength = edge_patch.mean(dim=(-1, -2))
        edge_strength = edge_strength - edge_strength.amin(dim=(2, 3), keepdim=True)
        denom = edge_strength.amax(dim=(2, 3), keepdim=True) + 1e-6
        edge_norm = edge_strength / denom
        return edge_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size
        x_pad, H0, W0, hp, wp = self._pad_to_multiple(x, P)
        B, C, H, W = x_pad.shape

        h_tiles = H // P
        w_tiles = W // P
        x_patch = (
            x_pad.view(B, C, h_tiles, P, w_tiles, P).permute(0, 1, 2, 4, 3, 5).contiguous()
        )
        x_fft = torch.fft.rfft2(x_patch.float())
        mask = self._radial_mask(C, device=x.device, dtype=x.dtype)
        mask = mask.to(device=x_fft.device, dtype=x_fft.real.dtype)

        if self.use_edge_enhance:
            x_mean = x.mean(dim=1, keepdim=True)
            edge_norm = self._edge_strength_per_patch(x_mean, h_tiles, w_tiles, (hp, wp))
            edge_coeff = 1.0 + self.edge_boost * edge_norm
            edge_coeff = edge_coeff.unsqueeze(-1).unsqueeze(-1).to(device=x_fft.device, dtype=x_fft.real.dtype)
            hf_mask = self.high_freq_mask.to(device=x_fft.device, dtype=x_fft.real.dtype)
            hf_mask = hf_mask.view(1, 1, 1, 1, P, P // 2 + 1)
            scaling = 1.0 + (edge_coeff - 1.0) * hf_mask
            x_fft = x_fft * mask * scaling
        else:
            x_fft = x_fft * mask

        x_patch = torch.fft.irfft2(x_fft, s=(P, P))
        x_rec = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        if hp or wp:
            x_rec = x_rec[:, :, :H0, :W0]
        return x_rec

class FourierUnitLite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, fu_kernel: int = 1, use_only_freq: bool = False, fft_norm: str = 'ortho', bias: bool = False):
        super().__init__()
        self.use_only_freq = use_only_freq
        self.fft_norm = fft_norm
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=fu_kernel, padding=fu_kernel // 2, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dims = (-2,) if self.use_only_freq else (-2, -1)
        s = (H,) if self.use_only_freq else (H, W)
        X = torch.fft.rfftn(x, dim=dims, norm=self.fft_norm)
        X2 = torch.stack((X.real, X.imag), dim=2).permute(0, 1, 2, 3, 4)  # (B, C, 2, ...)
        X2 = X2.reshape(B, C * 2, *X2.shape[-2:])
        Y = self.act(self.bn(self.conv(X2)))
        Y = Y.view(B, -1, 2, *Y.shape[-2:]).permute(0, 1, 3, 4, 2)
        Yc = torch.complex(Y[..., 0], Y[..., 1])
        y = torch.fft.irfftn(Yc, s=s, dim=dims, norm=self.fft_norm)
        return y


class SpectralTransformLite(nn.Module):
    """FFC-like spectral transform (Conv1x1 -> FourierUnitLite -> Conv1x1).
    Keeps in/out channels the same and is drop-in as a residual block.
    """
    def __init__(self, channels: int, fu_kernel: int = 1, use_only_freq: bool = False, fft_norm: str = 'ortho', bias: bool = False):
        super().__init__()
        half = max(1, channels // 2)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, half, kernel_size=1, bias=bias),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnitLite(half, half, fu_kernel=fu_kernel, use_only_freq=use_only_freq, fft_norm=fft_norm, bias=bias)
        self.post = nn.Conv2d(half, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = self.fu(y)
        y = self.post(y)
        return y

class FFTformerDFFNLite(nn.Module):
    """DFFN-like block from FFTformer adapted as a drop-in residual module.

    Structure (matching FFTformer DFFN order):
      project_in (C -> 2H)
      rFFT over PxxP non-overlapping patches with learnable per-(channel,frequency) mask
      depthwise conv over 2H channels, split -> GELU(x1) * x2
      project_out (H -> C)
    """

    def __init__(self, dim: int, expansion: float = 3.0, patch_size: int = 8, bias: bool = False):
        super().__init__()
        assert patch_size > 0 and patch_size % 2 == 0, "patch_size must be even and > 0"
        self.dim = dim
        self.hidden = int(dim * expansion)
        self.patch_size = patch_size

        self.project_in = nn.Conv2d(dim, self.hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(self.hidden * 2, self.hidden * 2, kernel_size=3, stride=1, padding=1,
                                groups=self.hidden * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((self.hidden * 2, 1, 1, patch_size, patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(self.hidden, dim, kernel_size=1, bias=bias)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        B, C, H, W = x.shape
        hp = (multiple - H % multiple) % multiple
        wp = (multiple - W % multiple) % multiple
        if hp or wp:
            x = F.pad(x, (0, wp, 0, hp), mode="reflect")
        return x, H, W, hp, wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size

        x = self.project_in(x)

        x_pad, H0, W0, hp, wp = self._pad_to_multiple(x, P)
        B2, C2, H2, W2 = x_pad.shape
        h_tiles = H2 // P
        w_tiles = W2 // P
        x_patch = x_pad.view(B2, C2, h_tiles, P, w_tiles, P).permute(0, 1, 2, 4, 3, 5).contiguous()
        x_fft = torch.fft.rfft2(x_patch.float())
        x_fft = x_fft * self.fft
        x_patch = torch.fft.irfft2(x_fft, s=(P, P))
        x = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B2, C2, H2, W2)
        if hp or wp:
            x = x[:, :, :H0, :W0]

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

