"""Unit tests for custom modules."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.carafe_module import CARAFE
from models.wiou_loss import WiseIoULoss


def test_carafe_forward_and_backward():
    print("[TEST] Testing CARAFE...")
    B, C, H, W = 2, 64, 16, 16
    scale = 2
    x = torch.randn(B, C, H, W, requires_grad=True)

    carafe = CARAFE(in_channels=C, out_channels=C, scale_factor=scale, up_kernel=5)
    out = carafe(x)

    expected_shape = (B, C, H * scale, W * scale)
    assert out.shape == expected_shape, f"CARAFE shape mismatch: got {out.shape}, expected {expected_shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "NaN found in input gradient"
    assert carafe.kernel_encoder.weight.grad is not None, "Encoder weight gradient is None"

    print(f"  --> Forward Shape: {out.shape} [PASS]")
    print(f"  --> Backward Gradient Flow: [PASS]")


def test_wiou_loss_versions():
    print("[TEST] Testing Wise-IoU Loss...")

    preds = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [15.0, 15.0, 55.0, 55.0],
        [100.0, 100.0, 150.0, 150.0]
    ], requires_grad=True)

    targets = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [10.0, 10.0, 50.0, 50.0],
        [10.0, 10.0, 50.0, 50.0]
    ])

    for v in [1, 2, 3]:
        criterion = WiseIoULoss(version=v)
        loss = criterion(preds, targets)

        assert loss.shape == (3,), f"WIoU v{v} shape mismatch: got {loss.shape}"
        assert loss[0].item() < 1e-4, f"WIoU v{v} perfect match loss should be ~0"

        total_loss = loss.sum()
        total_loss.backward(retain_graph=True)
        assert preds.grad is not None, f"WIoU v{v} gradient is None"
        assert not torch.isnan(preds.grad).any(), f"WIoU v{v} gradient contains NaN"

        print(f"  --> WIoU v{v} Loss: {loss.detach().numpy().round(4)} [PASS]")


def test_cuda_compatibility():
    print("[TEST] Testing CUDA Device Execution...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        carafe = CARAFE(in_channels=32, scale_factor=2).to(device)
        dummy_input = torch.randn(1, 32, 8, 8, device=device)
        out = carafe(dummy_input)
        assert out.is_cuda, "CARAFE output is not on CUDA device"
        print(f"  --> Executed on {torch.cuda.get_device_name(0)} [PASS]")
    else:
        print("  --> CUDA not available. Skipped.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    test_carafe_forward_and_backward()
    test_wiou_loss_versions()
    test_cuda_compatibility()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED")
    print("=" * 60)

