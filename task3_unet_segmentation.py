import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


NUM_SEG_CLASSES = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def init_wandb(args, run_name: str):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Missing dependency: wandb. Install with `pip install wandb`.") from exc
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or run_name,
        config=config,
    )


class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, size: int, train: bool, download: bool):
        self.ds = datasets.OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.size = size
        self.train = train
        self.image_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        image = transforms.functional.resize(
            image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR
        )
        mask = transforms.functional.resize(
            mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST
        )
        if self.train and torch.rand(()) < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        image = transforms.functional.to_tensor(image)
        image = self.image_norm(image)
        mask = transforms.functional.pil_to_tensor(mask).squeeze(0).long() - 1
        mask = mask.clamp(0, NUM_SEG_CLASSES - 1)
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, NUM_SEG_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(one_hot, dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def compute_miou(logits, target):
    pred = logits.argmax(1)
    ious = []
    for cls in range(NUM_SEG_CLASSES):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return sum(ious) / len(ious)


def make_loss(name):
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()
    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name == "ce_dice":
        return lambda logits, target: ce(logits, target) + dice(logits, target)
    raise ValueError(name)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_miou = 0.0
    total_batches = 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_miou += compute_miou(logits.detach(), masks)
            total_batches += 1
    return total_loss / total_batches, total_miou / total_batches


def main():
    parser = argparse.ArgumentParser(description="Task 3: U-Net segmentation from scratch")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default="ce_dice")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/task3"))
    parser.add_argument("--wandb", action="store_true", help="Log train/val curves to Weights & Biases")
    parser.add_argument("--wandb-project", default="cv-midterm-task3")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PetSegmentationDataset(args.data_root, "trainval", args.size, True, args.download)
    val_set = PetSegmentationDataset(args.data_root, "test", args.size, False, args.download)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    model = UNet(num_classes=NUM_SEG_CLASSES, base=args.base).to(device)
    criterion = make_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"unet_{args.loss}_e{args.epochs}_bs{args.batch_size}_lr{args.lr:g}"
    wandb_run = init_wandb(args, run_name)
    rows = []
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_miou = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        best_miou = max(best_miou, val_miou)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_miou": f"{train_miou:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_miou": f"{val_miou:.6f}",
                "best_val_miou": f"{best_miou:.6f}",
            }
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/mIoU": train_miou,
                    "val/loss": val_loss,
                    "val/mIoU": val_miou,
                    "val/best_mIoU": best_miou,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_miou={train_miou:.4f} val_miou={val_miou:.4f} best={best_miou:.4f}"
        )

    csv_path = args.out_dir / f"unet_{args.loss}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"model": model.state_dict(), "args": vars(args), "best_miou": best_miou}, args.out_dir / f"unet_{args.loss}.pt")
    print(f"Saved metrics and checkpoint to {args.out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
