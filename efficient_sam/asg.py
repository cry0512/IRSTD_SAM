import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AnisotropicSpectralGating(nn.Module):
    """
    各向异性光谱门控 (Anisotropic Spectral Gating, ASG) 模块
    
    该模块在频域中执行可学习的门控操作。与传统的径向门控不同，ASG 利用极坐标参数化，
    允许网络根据频率的方向（角度）和尺度（半径）联合调整频谱能量。
    
    这对于抑制具有特定方向性的红外背景杂波（如海浪、云层边缘）尤为有效。
    
    Args:
        dim (int): 输入特征图的通道数。
        h (int): 输入特征图的高度（用于构建频率网格）。
        w (int): 输入特征图的宽度。
        num_radial_bins (int): 极坐标滤波器的径向分辨率。默认为 64。
        num_angular_bins (int): 极坐标滤波器的角向分辨率。默认为 128。
    """
    def __init__(self, dim, h, w, num_radial_bins=64, num_angular_bins=128):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins

        # =====================================================================
        # 1. 定义极坐标系下的可学习参数 (The Learnable Filter)
        # =====================================================================
        # Shape: (1, C, Radial_Bins, Angular_Bins)
        # 初始化为 1.0，表示初始状态下为全通滤波器（Identity），不改变频谱。
        # 使用正态分布微扰打破对称性，有助于梯度流动。
        self.polar_weight = nn.Parameter(torch.ones(1, dim, num_radial_bins, num_angular_bins))
        nn.init.normal_(self.polar_weight, mean=1.0, std=0.01)

        # =====================================================================
        # 2. 预计算笛卡尔频域网格 (Cartesian Frequency Grid)
        # =====================================================================
        # 我们需要知道 rfft2 输出张量中每个像素点对应的物理频率。
        
        # 宽度方向 (kx): rfft 只保留正频率
        kx = torch.fft.rfftfreq(w, d=1.0) 
        
        # 高度方向 (ky): fft 保留正负频率，顺序为 [0,..., 0.5, -0.5,..., -0]
        # 注意：我们需要将其通过 fftshift 调整为 [-0.5,..., 0,..., 0.5] 的自然顺序吗？
        # 不需要。我们只需要数值正确即可。fftshift 会打乱张量内存布局，增加开销。
        # 我们直接使用 fftfreq 生成的数值进行计算。
        ky = torch.fft.fftfreq(h, d=1.0)
        
        # 生成网格 (Meshgrid)
        # grid_y shape: (H, W//2 + 1), grid_x shape: (H, W//2 + 1)
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing='ij')

        # =====================================================================
        # 3. 笛卡尔坐标转极坐标 (Cartesian -> Polar)
        # =====================================================================
        # 半径 r
        r = torch.sqrt(grid_x**2 + grid_y**2)
        
        # 角度 theta, range [-pi, pi]
        # 由于 kx >= 0 (rfft性质), theta 实际范围是 [-pi/2, pi/2]
        theta = torch.atan2(grid_y, grid_x)

        # =====================================================================
        # 4. 归一化坐标以适应 grid_sample (Normalize for Sampling)
        # =====================================================================
        # grid_sample 要求坐标范围在 [-1, 1]。
        
        # 半径归一化:
        # 最高频率 (Nyquist) 为 0.5。我们将 0.5 映射到 1.0。
        # r_norm = (r / 0.5) * 2 - 1  => r * 4 - 1
        # 范围 [0, 0.5] -> [-1, 1]。对于 r > 0.5 的部分（角落），r_norm > 1，
        # 将由 padding_mode='border' 处理，即复用边缘的高频权重。
        r_norm = (r / 0.5) * 2.0 - 1.0
        
        # 角度归一化:
        # theta 范围 [-pi/2, pi/2]。
        # 我们将其映射到 [-1, 1]。
        theta_norm = theta / (math.pi / 2.0)

        # 堆叠为 grid_sample 需要的格式 (N, H, W, 2)
        # 最后一个维度是 (x, y)，对应于 (theta, r) 或 (r, theta)。
        # grid_sample 的 grid[..., 0] 是 x (W维度), grid[..., 1] 是 y (H维度)。
        # 在我们的 polar_weight 中，W维度是 Angular，H维度是 Radial。
        # 所以 grid 应该是 (theta, r)。
        self.sampling_grid = torch.stack([theta_norm, r_norm], dim=-1).unsqueeze(0)
        
        # 将 grid 注册为 buffer，这样它会随模型保存和移动设备，但不是可训练参数
        self.register_buffer('grid', self.sampling_grid)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入特征图，形状为 (B, C, H, W)
        Returns:
            Tensor: 门控后的特征图，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 快速傅里叶变换 (RFFT2)
        # 输出形状: (B, C, H, W//2 + 1)
        # 使用 ortho 范数以保持能量守恒
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 生成频域掩膜 (Generate Spectral Mask)
        # 利用 grid_sample 从极坐标参数中采样出当前分辨率下的笛卡尔掩膜
        # 输入: polar_weight (1, C, Rad_Res, Ang_Res)
        # 网格: grid (1, H, W_fft, 2)
        # 输出: mask (1, C, H, W_fft)
        
        # 扩展 grid 以匹配 Batch 大小 (虽然 grid_sample 支持广播，但显式扩展更安全)
        grid_batch = self.grid.expand(B, -1, -1, -1)
        weight = self.polar_weight
        if weight.size(0) != B:
            weight = weight.expand(B, -1, -1, -1)
        
        # 采样模式：
        # mode='bilinear': 保证梯度平滑回传
        # padding_mode='border': 对于 r > 0.5 的高频角落，复用最外圈的权重
        # align_corners=True: 几何对齐
        mask = F.grid_sample(
            weight,
            grid_batch, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # 3. 门控操作 (Gating)
        # 使用 Sigmoid 激活函数将掩膜值约束在 (0, 1) 之间，起到衰减作用。
        # 如果希望允许信号增强，可以移除 Sigmoid 或使用其他激活函数。
        # 对于 SIRST 任务，背景抑制是主要目标，Sigmoid 较为合适。
        gate = torch.sigmoid(mask)
        
        # 执行复数乘法（广播机制：gate 作用于实部和虚部）
        x_gated = x_fft * gate
        
        # 4. 逆傅里叶变换 (IRFFT2)
        # 恢复到空间域，指定输出尺寸以处理奇偶性
        x_out = torch.fft.irfft2(x_gated, s=(H, W), norm='ortho')
        
        # 5. 残差连接 (Residual Connection)
        # 允许网络学习“修正量”，有助于训练稳定性
        return x + x_out

    def extra_repr(self):
        return f"dim={self.dim}, h={self.h}, w={self.w}, bins=({self.num_radial_bins}, {self.num_angular_bins})"


class AnisotropicSpectralGating2(nn.Module):
    def __init__(self, dim, h, w, r_bins=64, theta_bins=128):
        """
        各向异性光谱门控 (ASG)
        Args:
            dim: 输入通道数
            h, w: 输入特征图的分辨率 (用于预计算网格)
            r_bins: 径向分辨率 (控制对不同频率尺度的敏感度)
            theta_bins: 角度分辨率 (控制对不同方向的敏感度, 关键参数)
        """
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        
        # 1. 定义极坐标下的可学习权重 (Batch=1, C, R, Theta)
        # 初始化为1 (全通)，添加微小噪声打破对称性
        self.polar_weight = nn.Parameter(torch.ones(1, dim, r_bins, theta_bins))
        nn.init.normal_(self.polar_weight, mean=1.0, std=0.02)
        
        # 2. 预计算采样网格 (缓存以节省计算)
        self.register_buffer('grid', self._build_grid(h, w))

    def _build_grid(self, h, w):
        """
        构建从笛卡尔频域到极坐标权重的映射网格
        """
        # rfft2 的频率范围
        # u (height): [0, 1,..., H/2, -H/2,..., -1] -> 归一化到 [-0.5, 0.5]
        # v (width): -> 归一化到 [0, 0.5] (因为是实数FFT，只取一半)
        
        u = torch.fft.fftfreq(h)  # shape (H,)
        v = torch.fft.rfftfreq(w) # shape (W//2 + 1,)
        
        # 生成网格 (H, W//2+1)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        
        # --- 核心变换：笛卡尔 -> 极坐标 ---
        # 1. 计算半径 r (归一化到 0~1, 对应 0~Nyquist)
        # r = sqrt(u^2 + v^2). 最大可能模长是 sqrt(0.5^2 + 0.5^2) ≈ 0.707
        r = torch.sqrt(grid_u**2 + grid_v**2)
        # 将 r 映射到 grid_sample 需要的 [-1, 1] 区间
        # 假设我们关注的最高频是 0.5 (Nyquist)，则 r=0.5 对应 grid=1
        r_norm = (r / 0.5) * 2.0 - 1.0 
        
        # 2. 计算角度 theta (范围 -pi ~ pi)
        # arctan2(y, x) -> arctan2(u, v) 注意坐标轴对应
        theta = torch.atan2(grid_u, grid_v)
        # 将 theta 映射到 grid_sample 需要的 [-1, 1] 区间
        # rfft 只保留了右半平面 (v>=0)，所以 theta 范围主要是 [-pi/2, pi/2]
        theta_norm = theta / (math.pi / 2) 

        # 3. 堆叠为 grid_sample 需要的 (x, y) 格式
        # 注意：grid_sample 的坐标顺序是 (x, y)，对应我们权重的 (theta, r)
        # grid shape: (1, H, W_half, 2)
        grid = torch.stack([theta_norm, r_norm], dim=-1).unsqueeze(0)
        
        return grid

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. FFT 变换 ---
        # 转换到频域, shape: (B, C, H, W//2+1)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # --- 2. 动态生成各向异性掩膜 ---
        # 如果输入尺寸变了 (如多尺度训练)，需要重新计算 grid
        if H!= self.h or W!= self.w:
            current_grid = self._build_grid(H, W).to(x.device).expand(B, -1, -1, -1)
        else:
            current_grid = self.grid.expand(B, -1, -1, -1)
            
        # 使用双线性插值从极坐标权重中采样
        # 这一步实现了：对于频域中的每个点 (u,v)，根据它的半径和角度，
        # 去查找 polar_weight 中对应的值。
        # padding_mode='border' 保证超过 Nyquist 的高频角沿用边界值
        weight = self.polar_weight
        if weight.size(0) != B:
            weight = weight.expand(B, -1, -1, -1)
        mask = F.grid_sample(weight, current_grid,
                             mode='bilinear', padding_mode='border', align_corners=True)
        
        # --- 3. 门控与残差 ---
        # Sigmoid 限制权重在 (0, 1) 之间作为滤波器
        mask = torch.sigmoid(mask)
        
        # 应用掩膜 (广播机制自动处理实部虚部)
        x_fft_gated = x_fft * mask
        
        # IFFT 还原
        x_out = torch.fft.irfft2(x_fft_gated, s=(H, W), norm='ortho')
        
        # 残差连接 (这是关键，保留原始信息，学习增强部分)
        return x + x_out
