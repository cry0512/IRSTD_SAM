# -*- coding: utf-8 -*-
import re

with open(r'E:\code\EfficientSAM-main\EfficientSAM-main\efficient_sam\asg.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: First ASG class forward method - add float32 conversion
old1 = """        B, C, H, W = x.shape
        
        # 1. 快速傅里叶变换 (RFFT2)
        # 输出形状: (B, C, H, W//2 + 1)
        # 使用 ortho 范数以保持能量守恒
        x_fft = torch.fft.rfft2(x, norm='ortho')"""

new1 = """        B, C, H, W = x.shape
        
        # 保存原始 dtype 并转换为 float32 以避免 ComplexHalf NaN 问题
        orig_dtype = x.dtype
        x_fp32 = x.float() if x.dtype == torch.float16 else x
        
        # 1. 快速傅里叶变换 (RFFT2)
        # 输出形状: (B, C, H, W//2 + 1)
        # 使用 ortho 范数以保持能量守恒
        x_fft = torch.fft.rfft2(x_fp32, norm='ortho')"""

content = content.replace(old1, new1)

# Fix 2: First ASG class return statement - restore dtype
old2 = """        # 5. 残差连接 (Residual Connection)
        # 允许网络学习"修正量"，有助于训练稳定性
        return x + x_out"""

new2 = """        # 5. 残差连接 (Residual Connection)
        # 允许网络学习"修正量"，有助于训练稳定性
        # 恢复原始数据类型
        if orig_dtype == torch.float16:
            x_out = x_out.half()
        return x + x_out"""

content = content.replace(old2, new2)

# Fix 3: Second ASG class (ASG2) forward method
old3 = """    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. FFT 变换 ---
        # 转换到频域, shape: (B, C, H, W//2+1)
        x_fft = torch.fft.rfft2(x, norm='ortho')"""

new3 = """    def forward(self, x):
        B, C, H, W = x.shape
        
        # 保存原始 dtype 并转换为 float32 以避免 ComplexHalf NaN 问题
        orig_dtype = x.dtype
        x_fp32 = x.float() if x.dtype == torch.float16 else x
        
        # --- 1. FFT 变换 ---
        # 转换到频域, shape: (B, C, H, W//2+1)
        x_fft = torch.fft.rfft2(x_fp32, norm='ortho')"""

content = content.replace(old3, new3)

# Fix 4: Second ASG class (ASG2) return statement
old4 = """        # 残差连接 (这是关键，保留原始信息，学习增强部分)
        return x + x_out"""

new4 = """        # 残差连接 (这是关键，保留原始信息，学习增强部分)
        # 恢复原始数据类型
        if orig_dtype == torch.float16:
            x_out = x_out.half()
        return x + x_out"""

content = content.replace(old4, new4)

with open(r'E:\code\EfficientSAM-main\EfficientSAM-main\efficient_sam\asg.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully patched asg.py')
