import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
import cv2
import os

# ==========================================
# 1. 从 NUDT-SIRST 数据集读取真实的红外局部 Patch
# ==========================================
# 选用 000001.png，目标在 (x=112, y=179)，Patch 对比度在 102 ~ 186 之间
# 这个目标不仅非常清晰，而且它具有极其典型的“点扩散响应”分布
img_path = r'E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUDT-SIRST\images\000001.png'
mask_path = r'E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUDT-SIRST\masks\000001.png'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 计算该图像中目标的质心
y_indices, x_indices = np.where(mask > 0)
center_x = int(np.mean(x_indices))
center_y = int(np.mean(y_indices))

# 提取局部 Patch
patch_size = 31 # 保持奇数以中心对齐
half_size = patch_size // 2
Z_patch = img[center_y - half_size : center_y + half_size + 1, 
              center_x - half_size : center_x + half_size + 1].astype(np.float32)

# 为了绘图好看，定义显示范围（extent）把像素坐标映射到浮点物理坐标
phys_extent = [-4, 4, -4, 4]

# ==========================================
# 2. 设置 IEEE 期刊风格的基础作图参数
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
})

fig = plt.figure(figsize=(16, 5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

# 共用的伪彩色映射表
cmap_theme = 'inferno' # 使用 viridis 或 inferno，inferno对比度更高

# ==========================================
# Subplot 1: 笛卡尔坐标系的冗余 (Cartesian Misalignment)
# ==========================================
ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
# 插入真实的红外背景
im1 = ax1.imshow(Z_patch, extent=phys_extent, origin='lower', cmap=cmap_theme)
ax1.set_title('(a) Cartesian Grid on Real IR Patch')
ax1.set_xticks([])
ax1.set_yticks([])

# 绘制 3x3 的方形网格 (模拟感受野窗口)
grid_size = 8/3 # phys_extent 的总长是8，分成3格
for i in range(-1, 2):
    for j in range(-1, 2):
        # 框的逻辑中心点
        cx = i * grid_size
        cy = j * grid_size
        rect = Rectangle((cx - grid_size/2, cy - grid_size/2), 
                         grid_size, grid_size, linewidth=2, edgecolor='white', facecolor='none', alpha=0.9)
        ax1.add_patch(rect)
        
        # 标出冗余/引入杂波的角落区域（实际像素并非只有1个点，而是一整个方块）
        if i != 0 and j != 0:
            ax1.scatter(cx, cy, color='red', marker='x', s=150, linewidths=3)

# 标注
ax1.text(2.2, 2.7, 'Clutter Noise\nin Corners', color='red', weight='bold', fontsize=12, ha='center',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# ==========================================
# Subplot 2: 角向-空间网格的契合 (ASG Alignment)
# ==========================================
ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
ax2.imshow(Z_patch, extent=phys_extent, origin='lower', cmap=cmap_theme)
ax2.set_title('(b) ASG Alignment with True PSF')
ax2.set_xticks([])
ax2.set_yticks([])

# 绘制同心圆 (Radial Bins - 匹配实际光斑大小)
for r in [0.8, 1.8, 2.8, 3.8]: # 根据 Z_patch 中实际的光斑能量衰减范围而定
    circle = Circle((0, 0), r, color='cyan', fill=False, linewidth=2, alpha=0.9, linestyle='--')
    ax2.add_patch(circle)

# 绘制放射线 (Angular Bins)
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx = 5 * np.cos(angle)
    dy = 5 * np.sin(angle)
    ax2.plot([0, dx], [0, dy], color='cyan', linewidth=1.5, alpha=0.6)

# 标注
ax2.text(0, -3.5, 'Radial Bins fit target contours\nrejecting outer clutter', color='cyan', weight='bold', fontsize=12, ha='center',
         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

# ==========================================
# Subplot 3: 极坐标系下的特征转换展开 (Unrolled Feature)
# ==========================================
ax3 = fig.add_subplot(gs[0, 2])

# 对真实的离散图像做极坐标采样展开 (Image Warping)
# cv2.warpPolar 能够将图像沿着给定中心变换为极坐标形式
img_size_actual = Z_patch.shape[0]
center = (img_size_actual // 2, img_size_actual // 2)
max_radius = img_size_actual // 2 + 1

# 使用 warpPolar 解卷绕真实图片
polar_img = cv2.warpPolar(Z_patch, (max_radius, 360), center, max_radius, cv2.INTER_CUBIC | cv2.WARP_POLAR_LINEAR)

# 绘制展开后的图
# Y轴是角度(0~360)，X轴是半径(r)
# 真实数据在拉伸后，能够极其显著地看出高量能只在左侧（r值小的地方）形成垂直的亮带结构！
im3 = ax3.imshow(polar_img, aspect='auto', cmap=cmap_theme, extent=[0, max_radius, 360, 0])
ax3.set_title('(c) Unrolled ASG Feature Representation')
ax3.set_ylabel(r'Angular $\theta$ (Degrees)')
ax3.set_xlabel(r'Radial Distance $r$ (Pixels)')

# 标注规律性：目标形成垂直带，杂波被推向右侧
ax3.axvline(x=max_radius*0.3, color='cyan', linestyle='--', linewidth=2)
ax3.text(max_radius*0.13, 180, 'Target Concentrated', rotation=90, color='cyan', weight='bold', fontsize=12, va='center')
ax3.text(max_radius*0.75, 40, 'Decoupled\nClutter', color='white', weight='bold', fontsize=12, ha='center')

plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(output_dir, 'asg_motivation_NUDTSIRST.pdf')
png_path = os.path.join(output_dir, 'asg_motivation_NUDTSIRST.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Figures saved successfully to:\n- {pdf_path}\n- {png_path}")
