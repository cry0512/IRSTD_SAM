import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import cv2
import os

# ==========================================
# 1. 从 NUDT-SIRST 数据集读取真实的红外图像
# ==========================================
img_path = r'E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUDT-SIRST\images\000001.png'
mask_path = r'E:\code\SIRST-5K-main\SIRST-5K-main\dataset\NUDT-SIRST\masks\000001.png'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 计算该图像中目标的质心 (利用真实的 Ground Truth Mask)
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

# 创建 1x4 的子图布局
fig = plt.figure(figsize=(22, 5.5))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1], wspace=0.15)

cmap_theme = 'inferno' 

# ==========================================
# Subplot 1: 完整的原始图像 + 放大框指示 (Full Image + Zoom Inset)
# ==========================================
ax0 = fig.add_subplot(gs[0, 0])
# 显示整张原图
ax0.imshow(img, cmap='gray')  # 原图通常用灰度展示最真实
ax0.set_title('(a) Full Infrared Image')
ax0.set_xticks([])
ax0.set_yticks([])

# 在原图上画一个红色的框，指示目标在哪里
rect_full = Rectangle((center_x - half_size, center_y - half_size), 
                      patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none')
ax0.add_patch(rect_full)

# 在左下角或合适位置画一个放大后的插图（Inset）来展示这个红框里的内容
# 使用 mpl_toolkits.axes_grid1.inset_locator
axins = zoomed_inset_axes(ax0, zoom=3.5, loc='lower left') 
axins.imshow(img, cmap='gray')
# 限制插图的显示范围到目标区域
axins.set_xlim(center_x - half_size - 5, center_x + half_size + 5)
axins.set_ylim(center_y + half_size + 5, center_y - half_size - 5) # Y轴翻转是因为图像坐标一般是上到下
axins.set_xticks([])
axins.set_yticks([])
# 加上红框边线
for spine in axins.spines.values():
    spine.set_edgecolor('red')
    spine.set_linewidth(2)

# 画连接线，将原图的红框和放大的插图连起来
mark_inset(ax0, axins, loc1=1, loc2=2, fc="none", ec="red", alpha=0.6)


# ==========================================
# Subplot 2: 笛卡尔坐标系的冗余 (Cartesian Misalignment)
# ==========================================
ax1 = fig.add_subplot(gs[0, 1], aspect='equal')
# 插入真实的红外背景 (伪彩色)
ax1.imshow(Z_patch, extent=phys_extent, origin='lower', cmap=cmap_theme)
ax1.set_title('(b) Cartesian Grid Misalignment')
ax1.set_xticks([])
ax1.set_yticks([])

# 绘制 3x3 的方形网格 (模拟感受野窗口)
grid_size = 8/3 
for i in range(-1, 2):
    for j in range(-1, 2):
        cx = i * grid_size
        cy = j * grid_size
        rect = Rectangle((cx - grid_size/2, cy - grid_size/2), 
                         grid_size, grid_size, linewidth=2, edgecolor='white', facecolor='none', alpha=0.9)
        ax1.add_patch(rect)
        
        # 标出冗余/引入杂波的角落
        if i != 0 and j != 0:
            ax1.scatter(cx, cy, color='red', marker='x', s=120, linewidths=2.5)

# ==========================================
# Subplot 3: 角向-空间网格的契合 (ASG Alignment)
# ==========================================
ax2 = fig.add_subplot(gs[0, 2], aspect='equal')
ax2.imshow(Z_patch, extent=phys_extent, origin='lower', cmap=cmap_theme)
ax2.set_title('(c) Proposed ASG Alignment')
ax2.set_xticks([])
ax2.set_yticks([])

# 绘制同心圆 (Radial Bins)
for r in [0.8, 1.8, 2.8, 3.8]: 
    circle = Circle((0, 0), r, color='cyan', fill=False, linewidth=2, alpha=0.9, linestyle='--')
    ax2.add_patch(circle)

# 绘制放射线 (Angular Bins)
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx = 5 * np.cos(angle)
    dy = 5 * np.sin(angle)
    ax2.plot([0, dx], [0, dy], color='cyan', linewidth=1.5, alpha=0.6)

# ==========================================
# Subplot 4: 极坐标系下的特征转换展开 (Unrolled Feature)
# ==========================================
ax3 = fig.add_subplot(gs[0, 3])

img_size_actual = Z_patch.shape[0]
center = (img_size_actual // 2, img_size_actual // 2)
max_radius = img_size_actual // 2 + 1

# 使用 warpPolar 解卷绕真实图片
polar_img = cv2.warpPolar(Z_patch, (max_radius, 360), center, max_radius, cv2.INTER_CUBIC | cv2.WARP_POLAR_LINEAR)

# 绘制展开后的图
im3 = ax3.imshow(polar_img, aspect='auto', cmap=cmap_theme, extent=[0, max_radius, 360, 0])
ax3.set_title('(d) Unrolled ASG Feature')
ax3.set_ylabel(r'Angular $\theta$ (Degrees)')
ax3.set_xlabel(r'Radial Distance $r$ (Pixels)')

# 标注规律性：目标形成垂直带，杂波被推向右侧
ax3.axvline(x=max_radius*0.3, color='cyan', linestyle='--', linewidth=2)
ax3.text(max_radius*0.12, 180, 'Target', rotation=90, color='cyan', weight='bold', fontsize=14, va='center')
ax3.text(max_radius*0.65, 180, 'Clutter', rotation=90, color='white', weight='bold', fontsize=14, va='center')


plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(output_dir, 'asg_motivation_NUDTSIRST_v2.pdf')
png_path = os.path.join(output_dir, 'asg_motivation_NUDTSIRST_v2.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Figures saved successfully to:\n- {pdf_path}\n- {png_path}")
