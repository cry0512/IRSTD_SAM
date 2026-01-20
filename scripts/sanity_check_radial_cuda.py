import sys
sys.path.insert(0, r'E:\code\EfficientSAM-main\EfficientSAM-main')

import torch
from efficient_sam.efficient_sam import build_efficient_sam
from efficient_sam.freq_modules import RadialFreqGate

if not torch.cuda.is_available():
    print('CUDA not available on this machine')
    raise SystemExit(0)

device = torch.device('cuda')

# Build model (no checkpoint)
model = build_efficient_sam(encoder_patch_embed_dim=192, encoder_num_heads=3, checkpoint=None)
model.eval().to(device)

# Attach only RadialFreqGate
try:
    dim = model.image_encoder.neck[0].out_channels
except Exception:
    dim = 256
model.image_encoder.radial_gate = RadialFreqGate(dim, patch_size=8, num_bins=6, channel_shared=True).to(device)

# Dummy input on CUDA
B = 2
H, W = 512, 768
images = torch.randn(B, 3, H, W, device=device)

# Encoder embeddings
with torch.no_grad():
    emb = model.get_image_embeddings(images)
print('Embeddings:', tuple(emb.shape))

# Predict masks with 1 query and 2 points
max_num_queries = 1
num_pts = 2
pts = torch.zeros(B, max_num_queries, num_pts, 2, device=device)
pts[...,0] = torch.randint(0, W, (B, max_num_queries, num_pts), device=device)
pts[...,1] = torch.randint(0, H, (B, max_num_queries, num_pts), device=device)
labels = torch.ones(B, max_num_queries, num_pts, device=device)

with torch.no_grad():
    masks, iou_pred = model.predict_masks(emb, pts, labels, multimask_output=False, input_h=H, input_w=W, output_h=H, output_w=W)
print('Masks:', tuple(masks.shape), ' IOU:', tuple(iou_pred.shape))
print('Radial-only CUDA sanity-check OK')
