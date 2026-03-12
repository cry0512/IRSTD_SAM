import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 1. 模拟红外目标的高斯生成函数
# ==========================================
def gaussian_2d(x, y, x0=0, y0=0, sigma_x=1.5, sigma_y=1.5, A=1.0):
    return A * np.exp(-0.5 * (((x - x0) / sigma_x)**2 + ((y - y0) / sigma_y)**2))

# 创建坐标系
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x, y)
Z = gaussian_2d(X, Y)

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

# ==========================================
# Subplot 1: 笛卡尔坐标系的冗余 (Cartesian Misalignment)
# ==========================================
ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
ax1.imshow(Z, extent=[-4, 4, -4, 4], origin='lower', cmap='inferno')
ax1.set_title('(a) Cartesian Grid Mismatch')
ax1.set_xticks([])
ax1.set_yticks([])

# 绘制 3x3 的方形网格
grid_size = 2
for i in range(-1, 2):
    for j in range(-1, 2):
        rect = Rectangle((i*grid_size - grid_size/2, j*grid_size - grid_size/2), 
                         grid_size, grid_size, linewidth=1.5, edgecolor='black', facecolor='none', alpha=0.8)
        ax1.add_patch(rect)
        
        # 标出冗余的角落
        if i != 0 and j != 0:
            ax1.scatter(i*grid_size, j*grid_size, color='red', marker='x', s=100, linewidths=2)
            
# 添加文字标注
ax1.text(2, 2.5, 'Redundant\nNoisy Corners', color='red', weight='bold', fontsize=12, ha='center')

# ==========================================
# Subplot 2: 角向-空间网格的契合 (ASG Alignment)
# ==========================================
ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
ax2.imshow(Z, extent=[-4, 4, -4, 4], origin='lower', cmap='inferno')
ax2.set_title('(b) Angular-Spatial Grid (ASG)')
ax2.set_xticks([])
ax2.set_yticks([])

# 绘制同心圆 (Radial Bins)
for r in [0.5, 1.5, 2.5, 3.5]:
    circle = Circle((0, 0), r, color='cyan', fill=False, linewidth=1.5, alpha=0.8, linestyle='--')
    ax2.add_patch(circle)

# 绘制放射线 (Angular Bins)
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx = 4 * np.cos(angle)
    dy = 4 * np.sin(angle)
    ax2.plot([0, dx], [0, dy], color='cyan', linewidth=1, alpha=0.6)

# 等高线完美契合标注
ax2.text(0, -3.2, 'Contours align w/ Radial Bins', color='cyan', weight='bold', fontsize=12, ha='center')

# ==========================================
# Subplot 3: 极坐标系下的特征转换展开 (Unrolled Feature)
# ==========================================
ax3 = fig.add_subplot(gs[0, 2])

# 转换到 r-theta 空间
theta = np.linspace(0, 2*np.pi, 128)
r = np.linspace(0, 4, 64)
Theta, R = np.meshgrid(theta, r)
# 利用变换 X = R*cos(Theta), Y = R*sin(Theta) 计算高斯值
Z_polar = gaussian_2d(R * np.cos(Theta), R * np.sin(Theta))

# 绘制展开后的图
im3 = ax3.imshow(Z_polar, extent=[0, 360, 4, 0], origin='upper', cmap='inferno', aspect='auto')
ax3.set_title('(c) Unrolled ASG Feature Matrix')
ax3.set_xlabel(r'Angular $\theta$ (Degrees)')
ax3.set_ylabel(r'Radial Distance $r$')

# 标注规整性
ax3.axhline(y=1.5, color='cyan', linestyle='--', linewidth=1.5)
ax3.text(180, 1.2, 'Highly Structured Gradient along r', color='cyan', weight='bold', fontsize=12, ha='center')

plt.tight_layout()

# Save the figure
output_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(output_dir, 'asg_motivation.pdf')
png_path = os.path.join(output_dir, 'asg_motivation.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Figures saved successfully to:\n- {pdf_path}\n- {png_path}")
