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


class AdaptiveFrequencyDecomposition(nn.Module):
    """
    自适应频率分解模块 (Adaptive Frequency Decomposition, AFD) V2
    
    核心思想:
    1. 将特征图转换到频域
    2. 使用可学习网络预测每张图像的最优频率切分点
    3. 根据切分点分离高频和低频分量
    4. 使用可学习增益参数处理高低频后融合
    
    V2改进:
    - 将固定的low_ratio/high_ratio改为可学习的nn.Parameter
    - 增益初始化为1.0，避免初始阶段过度抑制/增强
    - 添加可选的通道级增益调制
    
    Args:
        dim (int): 输入特征通道数
        patch_size (int): FFT处理的patch大小，默认8
        num_cutoff_bins (int): 切分点预测的离散化粒度，默认16
        low_enhance_ratio (float): 低频增益初始值，默认1.0 (V2中仅作初始化用)
        high_enhance_ratio (float): 高频增益初始值，默认1.0 (V2中仅作初始化用)
        learnable_gains (bool): 是否使用可学习增益，默认True
        channel_wise_gains (bool): 是否使用通道级独立增益，默认False
    """
    
    def __init__(
        self,
        dim: int,
        patch_size: int = 8,
        num_cutoff_bins: int = 16,
        low_enhance_ratio: float = 1.0,
        high_enhance_ratio: float = 1.0,
        learnable_gains: bool = True,
        channel_wise_gains: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_bins = num_cutoff_bins
        self.learnable_gains = learnable_gains
        self.channel_wise_gains = channel_wise_gains
        
        # ========== 1. 可学习的频率增益参数 ==========
        if learnable_gains:
            if channel_wise_gains:
                # 每个通道独立的增益
                self.low_gain = nn.Parameter(torch.ones(dim) * low_enhance_ratio)
                self.high_gain = nn.Parameter(torch.ones(dim) * high_enhance_ratio)
            else:
                # 全局共享的增益
                self.low_gain = nn.Parameter(torch.tensor(low_enhance_ratio))
                self.high_gain = nn.Parameter(torch.tensor(high_enhance_ratio))
        else:
            # 保持向后兼容：固定增益
            self.register_buffer('low_gain', torch.tensor(low_enhance_ratio))
            self.register_buffer('high_gain', torch.tensor(high_enhance_ratio))
        
        # 保存原始值用于 extra_repr
        self._init_low_ratio = low_enhance_ratio
        self._init_high_ratio = high_enhance_ratio
        
        # ========== 2. 切分点预测网络 ==========
        # 根据全局特征预测最优的高低频切分点
        self.cutoff_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_cutoff_bins),
            nn.Softmax(dim=-1),
        )
        
        # ========== 3. 高频增强分支 ==========
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
        
        # ========== 4. 低频处理分支 ==========
        self.low_freq_process = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
        
        # ========== 5. 融合门控 ==========
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid(),
        )
        
        # ========== 5. 预计算径向频率模板 ==========
        self._precompute_radial_template()
    
    def _precompute_radial_template(self):
        """预计算归一化径向频率模板"""
        ps = self.patch_size
        fy = torch.fft.fftfreq(ps)
        fx = torch.fft.rfftfreq(ps)
        grid_fy, grid_fx = torch.meshgrid(fy, fx, indexing='ij')
        radial_dist = torch.sqrt(grid_fy ** 2 + grid_fx ** 2)
        radial_dist = radial_dist / (radial_dist.max() + 1e-8)
        bin_indices = (radial_dist * (self.num_bins - 1)).long()
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        self.register_buffer('radial_template', radial_dist)
        self.register_buffer('bin_indices', bin_indices)
    
    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (pad_h, pad_w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        
        # Step 1: 预测自适应切分点
        cutoff_probs = self.cutoff_predictor(x)
        bin_centers = torch.linspace(0, 1, self.num_bins, device=x.device)
        weighted_cutoff = (cutoff_probs * bin_centers.unsqueeze(0)).sum(dim=1)
        
        # Step 2: 填充并分块
        x_padded, pad_hw = self._pad_to_multiple(x, self.patch_size)
        Bp, Cp, Hp, Wp = x_padded.shape
        ps = self.patch_size
        n_h = Hp // ps
        n_w = Wp // ps
        
        x_patches = x_padded.reshape(B, C, n_h, ps, n_w, ps)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, ps, ps)
        
        # Step 3: FFT分解
        x_freq = torch.fft.rfft2(x_patches, norm='ortho')
        
        # Step 4: 自适应高低频分离
        cutoff_expanded = weighted_cutoff.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cutoff_expanded = cutoff_expanded.repeat(1, n_h * n_w, 1, 1).reshape(-1, 1, 1, 1)
        radial = self.radial_template.unsqueeze(0).unsqueeze(0)
        
        sharpness = 10.0
        high_mask = torch.sigmoid(sharpness * (radial - cutoff_expanded))
        low_mask = 1.0 - high_mask
        
        x_freq_high = x_freq * high_mask
        x_freq_low = x_freq * low_mask
        
        # Step 5: 逆FFT回到空域
        x_high = torch.fft.irfft2(x_freq_high, s=(ps, ps), norm='ortho')
        x_low = torch.fft.irfft2(x_freq_low, s=(ps, ps), norm='ortho')
        
        x_high = x_high.reshape(B, n_h, n_w, C, ps, ps)
        x_high = x_high.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        x_low = x_low.reshape(B, n_h, n_w, C, ps, ps)
        x_low = x_low.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        
        if pad_hw[0] > 0 or pad_hw[1] > 0:
            x_high = x_high[:, :, :H, :W]
            x_low = x_low[:, :, :H, :W]
        
        # Step 6: 分支处理 - 使用可学习增益
        x_high_enhanced = self.high_freq_enhance(x_high)
        x_low_processed = self.low_freq_process(x_low)
        
        # 应用可学习增益
        if self.channel_wise_gains:
            # 通道级增益: [C] -> [1, C, 1, 1]
            high_gain = self.high_gain.view(1, -1, 1, 1)
            low_gain = self.low_gain.view(1, -1, 1, 1)
        else:
            # 全局增益: scalar
            high_gain = self.high_gain
            low_gain = self.low_gain
        
        x_high_enhanced = x_high_enhanced * high_gain
        x_low_processed = x_low_processed * low_gain
        
        # Step 7: 自适应融合
        combined = torch.cat([x_high_enhanced, x_low_processed], dim=1)
        gate = self.fusion_gate(combined)
        output = gate * x_high_enhanced + (1 - gate) * x_low_processed
        
        return output + identity
    
    def extra_repr(self) -> str:
        if self.learnable_gains:
            low_val = self.low_gain.mean().item() if self.channel_wise_gains else self.low_gain.item()
            high_val = self.high_gain.mean().item() if self.channel_wise_gains else self.high_gain.item()
            gain_info = f'low_gain={low_val:.3f}(learnable), high_gain={high_val:.3f}(learnable)'
        else:
            gain_info = f'low_gain={self._init_low_ratio}, high_gain={self._init_high_ratio}(fixed)'
        return (f'dim={self.dim}, patch_size={self.patch_size}, '
                f'num_bins={self.num_bins}, {gain_info}')


class MultiScaleFrequencyEnhancement(nn.Module):
    """
    多尺度频率增强模块 (Multi-Scale Frequency Enhancement, MSFE)
    
    核心思想:
    1. 使用不同的 patch_size (4, 8, 16) 做 FFT 频率分解
    2. 小 patch (4) 捕获局部高频细节（边缘、纹理）
    3. 中 patch (8) 捕获中频信息
    4. 大 patch (16) 捕获全局低频结构（形状、轮廓）
    5. 每个尺度有独立的可学习频率增益
    6. 使用注意力机制自适应融合不同尺度的输出
    
    Args:
        dim (int): 输入特征通道数
        patch_sizes (tuple): 不同尺度的 patch 大小列表，默认 (4, 8, 16)
        num_radial_bins (int): 径向频率 bins 数量，默认 8
        fusion_method (str): 融合方法，'attention' 或 'concat'，默认 'attention'
    """
    
    def __init__(
        self,
        dim: int,
        patch_sizes: tuple = (4, 8, 16),
        num_radial_bins: int = 8,
        fusion_method: str = 'attention',
    ):
        super().__init__()
        self.dim = dim
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)
        self.num_radial_bins = num_radial_bins
        self.fusion_method = fusion_method
        
        # ========== 1. 每个尺度独立的可学习径向频率增益 ==========
        self.radial_gains = nn.ParameterList([
            nn.Parameter(torch.ones(num_radial_bins)) for _ in patch_sizes
        ])
        
        # ========== 2. 预计算每个尺度的径向频率模板 ==========
        for i, ps in enumerate(patch_sizes):
            self._precompute_radial_template(ps, i)
        
        # ========== 3. 融合模块 ==========
        if fusion_method == 'attention':
            # 注意力融合: 根据输入自适应选择尺度权重
            self.attention_fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 4, self.num_scales),
                nn.Softmax(dim=-1),
            )
        elif fusion_method == 'concat':
            # Concat 融合: 拼接后用 1x1 conv 压缩
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(dim * self.num_scales, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
        
        # ========== 4. 输出投影 ==========
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
    
    def _precompute_radial_template(self, patch_size: int, scale_idx: int):
        """预计算指定 patch_size 的归一化径向频率模板"""
        ps = patch_size
        fy = torch.fft.fftfreq(ps)
        fx = torch.fft.rfftfreq(ps)
        grid_fy, grid_fx = torch.meshgrid(fy, fx, indexing='ij')
        radial_dist = torch.sqrt(grid_fy ** 2 + grid_fx ** 2)
        radial_dist = radial_dist / (radial_dist.max() + 1e-8)
        
        # 将径向距离映射到 bin 索引
        bin_indices = (radial_dist * (self.num_radial_bins - 1)).long()
        bin_indices = bin_indices.clamp(0, self.num_radial_bins - 1)
        
        self.register_buffer(f'radial_dist_{scale_idx}', radial_dist)
        self.register_buffer(f'bin_indices_{scale_idx}', bin_indices)
    
    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (pad_h, pad_w)
    
    def _process_single_scale(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """处理单个尺度的频率增强"""
        B, C, H, W = x.shape
        ps = self.patch_sizes[scale_idx]
        
        # 填充到 patch_size 的倍数
        x_padded, pad_hw = self._pad_to_multiple(x, ps)
        Bp, Cp, Hp, Wp = x_padded.shape
        n_h = Hp // ps
        n_w = Wp // ps
        
        # 重排为 patches: [B*n_h*n_w, C, ps, ps]
        x_patches = x_padded.reshape(B, C, n_h, ps, n_w, ps)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, ps, ps)
        
        # FFT
        x_freq = torch.fft.rfft2(x_patches.float(), norm='ortho')
        
        # 获取该尺度的增益和 bin 索引
        gains = self.radial_gains[scale_idx]  # [num_bins]
        bin_indices = getattr(self, f'bin_indices_{scale_idx}')  # [ps, ps//2+1]
        
        # 根据 bin 索引查找增益
        freq_gains = gains[bin_indices]  # [ps, ps//2+1]
        freq_gains = freq_gains.unsqueeze(0).unsqueeze(0)  # [1, 1, ps, ps//2+1]
        
        # 应用增益
        x_freq = x_freq * freq_gains
        
        # 逆 FFT
        x_spatial = torch.fft.irfft2(x_freq, s=(ps, ps), norm='ortho')
        
        # 重排回原始形状
        x_spatial = x_spatial.reshape(B, n_h, n_w, C, ps, ps)
        x_spatial = x_spatial.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        
        # 裁剪掉 padding
        if pad_hw[0] > 0 or pad_hw[1] > 0:
            x_spatial = x_spatial[:, :, :H, :W]
        
        return x_spatial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        
        # 处理每个尺度
        scale_outputs = []
        for i in range(self.num_scales):
            out_i = self._process_single_scale(x, i)
            scale_outputs.append(out_i)
        
        # 融合
        if self.fusion_method == 'attention':
            # 计算注意力权重
            attn_weights = self.attention_fc(x)  # [B, num_scales]
            
            # 加权融合
            output = torch.zeros_like(x)
            for i, out_i in enumerate(scale_outputs):
                weight_i = attn_weights[:, i].view(B, 1, 1, 1)
                output = output + weight_i * out_i
        
        elif self.fusion_method == 'concat':
            # 拼接融合
            concat = torch.cat(scale_outputs, dim=1)  # [B, C*num_scales, H, W]
            output = self.fusion_conv(concat)
        
        else:
            # 简单平均
            output = torch.stack(scale_outputs, dim=0).mean(dim=0)
        
        # 输出投影
        output = self.output_proj(output)
        
        return output + identity
    
    def extra_repr(self) -> str:
        gains_info = []
        for i, ps in enumerate(self.patch_sizes):
            g = self.radial_gains[i]
            gains_info.append(f'ps{ps}:[{g.min().item():.2f},{g.max().item():.2f}]')
        return (f'dim={self.dim}, patch_sizes={self.patch_sizes}, '
                f'num_radial_bins={self.num_radial_bins}, '
                f'fusion={self.fusion_method}, gains={{{", ".join(gains_info)}}}')
