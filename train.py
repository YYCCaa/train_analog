"""
Train a lightweight U-Net to predict a heatmap of the dial pointer tip.

Architecture: custom encoder-decoder with skip connections (~500K params).
    Encoder: 4 blocks  32 → 64 → 128 → 256  (each: Conv-BN-ReLU-Conv-BN-ReLU + MaxPool)
    Decoder: 4 blocks  256 → 128 → 64 → 32  (bilinear upsample + skip-cat + Conv-BN-ReLU x2)
    Head:    1×1 conv → 1-channel heatmap (sigmoid)

Target: 2-D Gaussian heatmap (sigma=5 px) centred on the pointer tip,
        generated on-the-fly from (tip_x, tip_y) in labels.csv.

Input:   256×256 RGB image  →  normalised to [0,1]
Output:  256×256 single-channel heatmap
Loss:    MSE
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

# ── dataset ─────────────────────────────────────────────────────────────

GAUSSIAN_SIGMA = 5.0


def make_heatmap(tip_x, tip_y, size, sigma=GAUSSIAN_SIGMA):
    """Generate a 2-D Gaussian heatmap centred on (tip_x, tip_y)."""
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)  # yy = row index, xx = col index
    heatmap = np.exp(-((xx - tip_x) ** 2 + (yy - tip_y) ** 2) / (2 * sigma ** 2))
    return heatmap  # peak = 1.0 at (tip_x, tip_y)


class DialDataset(Dataset):
    """Loads synthetic dial images + generates Gaussian heatmap targets."""

    def __init__(self, data_dir, img_size=192):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []

        csv_path = os.path.join(data_dir, "labels.csv")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "tip_x": float(row["tip_x"]),
                    "tip_y": float(row["tip_y"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.data_dir, s["filename"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0      # H×W×3, [0,1]
        img = img.transpose(2, 0, 1)                        # 3×H×W

        heatmap = make_heatmap(s["tip_x"], s["tip_y"], self.img_size)  # H×W

        return (torch.from_numpy(img),
                torch.from_numpy(heatmap).unsqueeze(0))      # 1×H×W


# ── model ───────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TipHeatmapNet(nn.Module):
    """Lightweight U-Net for single-keypoint heatmap regression."""

    def __init__(self, in_channels=3, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder path
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        rev = list(reversed(features))
        self.upconvs.append(nn.ConvTranspose2d(features[-1] * 2, rev[0], 2, stride=2))
        self.decoders.append(ConvBlock(rev[0] * 2, rev[0]))
        for i in range(1, len(rev)):
            self.upconvs.append(nn.ConvTranspose2d(rev[i - 1], rev[i], 2, stride=2))
            self.decoders.append(ConvBlock(rev[i] * 2, rev[i]))

        # 1×1 head → sigmoid (heatmap in [0,1])
        self.head = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.head(x))


# ── training ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dist = 0.0
    count = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)
        running_loss += loss.item() * imgs.size(0)

        # Compute Euclidean distance between predicted and GT peak
        b = preds.size(0)
        for i in range(b):
            pred_map = preds[i, 0]
            gt_map = targets[i, 0]
            # argmax → (y, x)
            pred_flat = pred_map.view(-1)
            pred_idx = pred_flat.argmax().item()
            pred_y, pred_x = divmod(pred_idx, pred_map.size(1))

            gt_flat = gt_map.view(-1)
            gt_idx = gt_flat.argmax().item()
            gt_y, gt_x = divmod(gt_idx, gt_map.size(1))

            dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            total_dist += dist
            count += 1

    avg_loss = running_loss / len(loader.dataset)
    avg_dist = total_dist / max(count, 1)
    return avg_loss, avg_dist


def _update_sigma(val):
    global GAUSSIAN_SIGMA
    GAUSSIAN_SIGMA = val


def main():
    parser = argparse.ArgumentParser(description="Train tip-heatmap U-Net")
    parser.add_argument("--data-dir", default="synthetic_data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--save-path", default="tip_heatmap_net.pth")
    parser.add_argument("--sigma", type=float, default=GAUSSIAN_SIGMA)
    args = parser.parse_args()

    _update_sigma(args.sigma)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = DialDataset(args.data_dir)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {n_train}  Val: {n_val}")

    # Model
    model = TipHeatmapNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dist = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dist = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  "
              f"val_dist={val_dist:.2f}px  "
              f"lr={lr_now:.2e}")

        if val_dist < best_dist:
            best_dist = val_dist
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> saved best model (dist={best_dist:.2f}px)")

    print(f"\nTraining complete. Best val distance: {best_dist:.2f}px")
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
