"""
External SOTA Baseline: U-Net for Histopathology Multi-Class Tissue Segmentation.
Provides comparison benchmark required by Q1 journal reviewers:
Compares Proposed YOLOv8s-CARAFE-WIoU against standard medical semantic segmentation baseline (U-Net).
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net Architecture for Medical Image Segmentation."""
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits


class HistologyPatchDataset(Dataset):
    """Dataset loader for histology patches."""
    def __init__(self, img_dir: Path, lbl_dir: Path, img_size: int = 512):
        self.img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        self.lbl_dir = lbl_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Reconstruct mask from label
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)

        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        cls_id = int(parts[0])
                        pts = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                        pts[:, 0] *= self.img_size
                        pts[:, 1] *= self.img_size
                        cv2.fillPoly(mask, [pts.astype(np.int32)], cls_id)

        # Normalize image to [0, 1] and transpose to (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor


def train_unet_baseline(
    data_dir: str = "data/liver_primary/processed",
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "0"
):
    """Trains U-Net baseline on the histology dataset."""
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training U-Net Baseline on device: {dev}")

    dataset_path = Path(data_dir)
    train_dataset = HistologyPatchDataset(dataset_path / "images", dataset_path / "labels")
    if len(train_dataset) == 0:
        print("[WARN] No processed patches found. Run data/preprocessing/tiling.py first.")
        return

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = UNet(n_channels=3, n_classes=3).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(dev), masks.to(dev)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / max(len(loader), 1)
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save baseline weights
    out_weights = PROJECT_ROOT / "results" / "checkpoints" / "unet_baseline.pth"
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_weights))
    print(f"[SUCCESS] U-Net Baseline trained and saved to: {out_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Medical Segmentation Baseline")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    train_unet_baseline(epochs=args.epochs, batch_size=args.batch, device=args.device)
