import argparse
import csv
import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


NUM_CLASSES = 37
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


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


def _basicblock_forward_with_se(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.se(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out


def add_se_blocks(model: nn.Module, reduction: int = 16) -> nn.Module:
    """Attach SE blocks to every ResNet BasicBlock without relying on pretrained SE weights."""
    for module in model.modules():
        if isinstance(module, models.resnet.BasicBlock):
            module.se = SEBlock(module.bn2.num_features, reduction)
            module.forward = MethodType(_basicblock_forward_with_se, module)
    return model


def build_model(arch: str, pretrained: bool, attention: str) -> nn.Module:
    weights = None
    if pretrained:
        weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        }[arch]
    model = getattr(models, arch)(weights=weights)
    if attention == "se":
        model = add_se_blocks(model)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def make_loaders(data_root: Path, batch_size: int, workers: int, download: bool, pin_memory: bool):
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_set = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=download,
    )
    test_set = datasets.OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="category",
        transform=test_tf,
        download=download,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, log_interval: int = 20):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_seen += images.size(0)
            if training and log_interval > 0 and step % log_interval == 0:
                print(
                    f"  step {step:04d}/{len(loader)} "
                    f"loss={total_loss / total_seen:.4f} "
                    f"acc={total_correct / total_seen:.4f}",
                    flush=True,
                )
    return total_loss / total_seen, total_correct / total_seen


def save_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Task 1: Oxford-IIIT Pet fine-tuning")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--attention", choices=["none", "se"], default="none")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights")
    parser.add_argument("--random-init", action="store_true", help="Force random initialization")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/task1"))
    parser.add_argument("--wandb", action="store_true", help="Log train/val curves to Weights & Biases")
    parser.add_argument("--wandb-project", default="cv-midterm-task1")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if args.random_init:
        args.pretrained = False

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    if device.type == "cpu":
        print(
            "Warning: CUDA is not available. Training ResNet34 on CPU can be very slow; "
            "use --arch resnet18 --batch-size 4 for a quick smoke test.",
            flush=True,
        )
    train_loader, test_loader = make_loaders(
        args.data_root, args.batch_size, args.workers, args.download, pin_memory
    )
    model = build_model(args.arch, pretrained=args.pretrained, attention=args.attention).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = (
        f"{args.arch}_pre{int(args.pretrained)}_{args.attention}_"
        f"eh{args.epochs}_lh{args.lr_head:g}_lb{args.lr_backbone:g}"
    )
    wandb_run = init_wandb(args, run_name)
    rows = []
    best_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, args.log_interval
        )
        val_loss, val_acc = run_epoch(model, test_loader, criterion, device, None, 0)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "best_val_acc": f"{best_acc:.6f}",
        }
        rows.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/best_accuracy": best_acc,
                    "lr/backbone": optimizer.param_groups[0]["lr"],
                    "lr/head": optimizer.param_groups[1]["lr"],
                },
                step=epoch,
            )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}"
        )

    save_metrics(args.out_dir / f"{run_name}.csv", rows)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_acc": best_acc,
        },
        args.out_dir / f"{run_name}.pt",
    )
    print(f"Saved metrics and checkpoint to {args.out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
