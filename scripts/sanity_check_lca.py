"""
Sanity check for LCA-Prompt module.

Verifies:
- Import and instantiation
- Forward pass shapes (LCM, ASG bridge, Top-K extraction)
- LCA supervision loss backward
- Integration with EfficientSamHQ neck features
"""

import sys
import torch

sys.path.insert(0, r"E:\code\EfficientSAM-main\EfficientSAM-main")


def test_lcm_standalone():
    """Test DifferentiableLCM standalone."""
    from efficient_sam.lca_prompt import DifferentiableLCM

    lcm = DifferentiableLCM(scales=(3, 5, 9))
    images = torch.randn(2, 3, 256, 256)
    out = lcm(images)
    assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256) got {out.shape}"
    assert out.min() >= 0.0, f"LCM should be >= 0, got {out.min()}"
    assert out.max() <= 1.0 + 1e-4, f"LCM should be <= 1, got {out.max()}"
    print(f"[PASS] DifferentiableLCM: output shape={tuple(out.shape)}, "
          f"range=[{out.min():.4f}, {out.max():.4f}]")


def test_asg_bridge():
    """Test ASGContrastBridge."""
    from efficient_sam.lca_prompt import ASGContrastBridge

    bridge = ASGContrastBridge(neck_dim=256)
    lcm = torch.rand(2, 1, 256, 256)
    neck = torch.randn(2, 256, 16, 16)
    out = bridge(lcm, neck)
    assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256) got {out.shape}"
    print(f"[PASS] ASGContrastBridge: output shape={tuple(out.shape)}, "
          f"range=[{out.min():.4f}, {out.max():.4f}]")


def test_top_k_extractor():
    """Test SoftTopKExtractor."""
    from efficient_sam.lca_prompt import SoftTopKExtractor

    ext = SoftTopKExtractor(top_k=5, min_dist=8)
    contrast = torch.rand(2, 1, 256, 256)
    # Without GT
    coords, labels = ext(contrast, gt_mask=None)
    assert coords.shape == (2, 5, 2), f"Expected (2,5,2) got {coords.shape}"
    assert labels.shape == (2, 5), f"Expected (2,5) got {labels.shape}"
    print(f"[PASS] SoftTopKExtractor (no GT): coords={tuple(coords.shape)}, labels={tuple(labels.shape)}")

    # With GT
    gt = torch.zeros(2, 256, 256)
    gt[:, 100:110, 100:110] = 1.0
    coords, labels = ext(contrast, gt_mask=gt)
    assert coords.shape == (2, 5, 2), f"Expected (2,5,2) got {coords.shape}"
    print(f"[PASS] SoftTopKExtractor (with GT): coords={tuple(coords.shape)}, "
          f"labels range=[{labels.min():.0f}, {labels.max():.0f}]")


def test_lca_loss():
    """Test LCASupervisionLoss backward."""
    from efficient_sam.lca_prompt import LCASupervisionLoss

    loss_fn = LCASupervisionLoss()
    contrast = torch.rand(2, 1, 256, 256, requires_grad=True)
    gt = torch.zeros(2, 256, 256)
    gt[:, 100:110, 100:110] = 1.0
    loss = loss_fn(contrast, gt)
    loss.backward()
    assert contrast.grad is not None, "Gradient should flow through LCA loss"
    print(f"[PASS] LCASupervisionLoss: loss={loss.item():.4f}, grad OK")


def test_lca_generator():
    """Test full LCAPromptGenerator pipeline."""
    from efficient_sam.lca_prompt import LCAPromptGenerator

    gen = LCAPromptGenerator(
        scales=(3, 5, 9),
        top_k=5,
        min_dist=8,
        use_asg_bridge=True,
        neck_dim=256,
    )
    images = torch.randn(2, 3, 256, 256)
    neck = torch.randn(2, 256, 16, 16)
    gt = torch.zeros(2, 256, 256)
    gt[:, 100:110, 100:110] = 1.0

    # With GT (training mode)
    coords, labels, contrast, lca_loss = gen(images, neck_features=neck, gt_mask=gt)
    assert coords.shape == (2, 5, 2), f"Coords shape: {coords.shape}"
    assert labels.shape == (2, 5), f"Labels shape: {labels.shape}"
    assert contrast.shape == (2, 1, 256, 256), f"Contrast shape: {contrast.shape}"
    assert lca_loss is not None, "LCA loss should not be None with GT"
    lca_loss.backward()
    print(f"[PASS] LCAPromptGenerator (training): coords={tuple(coords.shape)}, "
          f"loss={lca_loss.item():.4f}, backward OK")

    # Without GT (inference mode)
    coords2, labels2, contrast2, lca_loss2 = gen(
        images, neck_features=neck, gt_mask=None
    )
    assert coords2.shape == (2, 5, 2)
    assert lca_loss2 is None
    print(f"[PASS] LCAPromptGenerator (inference): coords={tuple(coords2.shape)}, "
          f"loss=None (correct)")

    n_params = sum(p.numel() for p in gen.parameters())
    n_trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"[INFO] LCAPromptGenerator: total params={n_params:,}, trainable={n_trainable:,}")


def test_no_asg_bridge():
    """Test LCAPromptGenerator without ASG bridge."""
    from efficient_sam.lca_prompt import LCAPromptGenerator

    gen = LCAPromptGenerator(
        scales=(3, 5),
        top_k=3,
        use_asg_bridge=False,
    )
    images = torch.randn(2, 3, 128, 128)
    gt = torch.zeros(2, 128, 128)
    gt[:, 50:55, 50:55] = 1.0

    coords, labels, contrast, lca_loss = gen(images, neck_features=None, gt_mask=gt)
    assert coords.shape == (2, 3, 2)
    lca_loss.backward()
    print(f"[PASS] LCAPromptGenerator (no bridge): coords={tuple(coords.shape)}, loss={lca_loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("LCA-Prompt Sanity Check")
    print("=" * 60)
    test_lcm_standalone()
    test_asg_bridge()
    test_top_k_extractor()
    test_lca_loss()
    test_lca_generator()
    test_no_asg_bridge()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
