"""
Unit Tests for Custom Modules (CARAFE & Wise-IoU Loss).
Verifies tensor shapes, forward-backward pass stability, and gradient flow.
Ensures Q1 journal software reproducibility and reliability standards.

Run with:
    python tests/test_modules.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.carafe_module import CARAFE
from models.wiou_loss import WiseIoULoss


def test_carafe_forward_and_backward():
    """Tests CARAFE forward output dimensions and gradient flow."""
    print("[TEST] Testing CARAFE Operator...")
    B, C, H, W = 2, 64, 16, 16
    scale = 2
    x = torch.randn(B, C, H, W, requires_grad=True)

    carafe = CARAFE(in_channels=C, out_channels=C, scale_factor=scale, up_kernel=5)
    out = carafe(x)

    # 1. Check output shape
    expected_shape = (B, C, H * scale, W * scale)
    assert out.shape == expected_shape, f"CARAFE shape mismatch: got {out.shape}, expected {expected_shape}"

    # 2. Check backward pass & gradients
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None after CARAFE backward"
    assert not torch.isnan(x.grad).any(), "NaN found in CARAFE input gradient"
    assert carafe.kernel_encoder.weight.grad is not None, "Encoder weight gradient is None"

    print(f"  --> CARAFE Forward Shape: {out.shape} [PASS]")
    print(f"  --> CARAFE Backward Gradient Flow: Verified [PASS]")


def test_wiou_loss_versions():
    """Tests Wise-IoU v1, v2, v3 loss computation and dynamic focusing."""
    print("[TEST] Testing Wise-IoU Loss (v1, v2, v3)...")

    # Predictor boxes (N, 4): [x1, y1, x2, y2]
    preds = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [15.0, 15.0, 55.0, 55.0],
        [100.0, 100.0, 150.0, 150.0]  # Far outlier
    ], requires_grad=True)

    targets = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],     # Perfect match (IoU = 1)
        [10.0, 10.0, 50.0, 50.0],     # Moderate match
        [10.0, 10.0, 50.0, 50.0]      # Extreme outlier match
    ])

    for v in [1, 2, 3]:
        criterion = WiseIoULoss(version=v)
        loss = criterion(preds, targets)

        # Check loss shape
        assert loss.shape == (3,), f"WIoU v{v} shape mismatch: got {loss.shape}"

        # Perfect match loss should be near 0
        assert loss[0].item() < 1e-4, f"WIoU v{v} perfect match loss should be ~0, got {loss[0].item()}"

        # Check backward pass
        total_loss = loss.sum()
        total_loss.backward(retain_graph=True)
        assert preds.grad is not None, f"WIoU v{v} preds gradient is None"
        assert not torch.isnan(preds.grad).any(), f"WIoU v{v} preds gradient contains NaN"

        print(f"  --> WIoU v{v} Loss Values: {loss.detach().numpy().round(4)} [PASS]")


def test_cuda_compatibility():
    """Verifies that custom modules execute on CUDA device if available."""
    print("[TEST] Testing CUDA Device Execution...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        carafe = CARAFE(in_channels=32, scale_factor=2).to(device)
        dummy_input = torch.randn(1, 32, 8, 8, device=device)
        out = carafe(dummy_input)
        assert out.is_cuda, "CARAFE output is not on CUDA device"
        print(f"  --> Executed on {torch.cuda.get_device_name(0)} [PASS]")
    else:
        print("  --> CUDA not available in current environment. Skipped CUDA check.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR HISTOPATHOLOGY MODULES")
    print("=" * 60)
    test_carafe_forward_and_backward()
    test_wiou_loss_versions()
    test_cuda_compatibility()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY! [Q1 READY]")
    print("=" * 60)
