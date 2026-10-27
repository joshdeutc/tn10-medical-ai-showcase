"""
Entraînement SmallCNN sur CIFAR-10 en mode binaire (one-vs-all).

Banc d'essai "easy mode" pour valider toute la chaîne abstention/SGP sur des
données où le modèle peut effectivement atteindre un score de confiance
informatif (AUC ≥ 0.8 visé). Une fois ça validé, le même pipeline pourra être
appliqué aux mammographies sans douter de la mécanique.

Architecture identique au SmallCNN du notebook `intro_pytorch.qmd`, juste
re-paramétré avec `in_channels=3` (RGB) et `AdaptiveAvgPool2d((8,8))` pour
accepter des tailles d'image > 32×32.

Usage:
    python -m fine_tuning.train_smallcnn_cifar
    python -m fine_tuning.train_smallcnn_cifar --positive-class 0       # airplane vs all
    python -m fine_tuning.train_smallcnn_cifar --positive-class 5 --loss focal
    python -m fine_tuning.train_smallcnn_cifar --img-size 64 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from fine_tuning.config import PROJECT_ROOT, RANDOM_SEED
from fine_tuning.run_metadata import (
    format_duration,
    get_git_commit,
    make_run_dir,
    write_args_json,
    write_run_readme,
)

torch.backends.cudnn.benchmark = True

# ─── Hyperparamètres ─────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR = CHECKPOINT_DIR / "runs"
CIFAR_DIR = Path(PROJECT_ROOT) / "data" / "cifar10"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMG_SIZE = 96
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.3
EARLY_STOP_PATIENCE = 6
WARMUP_EPOCHS = 2
NUM_WORKERS = 4


# ─── Modèle ──────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """SmallCNN (notebook intro_pytorch.qmd) — version RGB."""

    def __init__(self, num_classes: int = 1, in_channels: int = 3, dropout: float = DROPOUT_P):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ─── Loss ────────────────────────────────────────────────────────────────────

class FocalLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ─── Dataset : CIFAR-10 binarisé ─────────────────────────────────────────────

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class CIFARBinary(Dataset):
    """Wrappe torchvision CIFAR10 et binarise les labels : 1 si classe == positive_class, sinon 0."""

    def __init__(self, root: Path, train: bool, positive_class: int, img_size: int, augment: bool):
        self.base = CIFAR10(root=str(root), train=train, download=False)
        self.positive_class = positive_class
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=img_size // 8, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, cls = self.base[idx]
        label = 1.0 if cls == self.positive_class else 0.0
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ─── Boucle d'entraînement ───────────────────────────────────────────────────

def train(
    positive_class: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    img_size: int,
    device: str,
    patience: int,
    loss_name: str,
) -> None:
    if loss_name not in {"bce", "bce-pos-weight", "focal"}:
        raise ValueError(f"loss_name doit être 'bce', 'bce-pos-weight' ou 'focal' (reçu: {loss_name})")
    if positive_class not in range(10):
        raise ValueError(f"positive_class doit être dans [0, 9] (reçu: {positive_class})")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    target = f"cifar_{CIFAR_CLASSES[positive_class]}_vs_all"
    model_tag = "smallcnn" if loss_name == "bce" else f"smallcnn-{loss_name}"
    run_dir = make_run_dir(RUNS_DIR, target, model_tag)
    ckpt_path = run_dir / "best.pt"
    logs_path = run_dir / "logs.json"
    roc_path = run_dir / "roc.png"

    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    train_ds = CIFARBinary(CIFAR_DIR, train=True, positive_class=positive_class, img_size=img_size, augment=True)
    val_ds = CIFARBinary(CIFAR_DIR, train=False, positive_class=positive_class, img_size=img_size, augment=False)

    n_pos_train = sum(1 for _, c in train_ds.base if c == positive_class)
    n_pos_val = sum(1 for _, c in val_ds.base if c == positive_class)

    print(f"Tâche : {target}")
    print(f"Run dir : {run_dir}")
    print(f"Train : {len(train_ds)} images  |  positifs ({CIFAR_CLASSES[positive_class]})={n_pos_train} "
          f"({100*n_pos_train/len(train_ds):.1f}%)")
    print(f"Val   : {len(val_ds)} images  |  positifs={n_pos_val} "
          f"({100*n_pos_val/len(val_ds):.1f}%)")
    print(f"Device: {device}  |  batch_size={batch_size}  |  img_size={img_size}x{img_size}")
    print(f"lr={lr}  |  weight_decay={weight_decay}  |  early-stop patience={patience}")
    print()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )

    model = SmallCNN(num_classes=1, in_channels=3, dropout=DROPOUT_P).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SmallCNN : {n_params:,} paramètres ({n_params/1e6:.2f} M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_iters = min(WARMUP_EPOCHS, max(1, epochs - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_iters)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_iters]
    )

    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        loss_desc = "BCEWithLogitsLoss"
    elif loss_name == "bce-pos-weight":
        pos_weight = torch.tensor(
            [(len(train_ds) - n_pos_train) / max(1, n_pos_train)], device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_desc = f"BCEWithLogitsLoss(pos_weight={pos_weight.item():.3f})"
    else:  # focal
        alpha = 1.0 - n_pos_train / len(train_ds)
        loss_fn = FocalLossBinary(alpha=alpha, gamma=2.0)
        loss_desc = f"FocalLossBinary(alpha={alpha:.3f}, gamma=2.0)"
    print(f"Loss : {loss_desc}")
    print()

    hyperparams = {
        "target": target,
        "positive_class": positive_class,
        "positive_class_name": CIFAR_CLASSES[positive_class],
        "model_arch": "SmallCNN (RGB, 2 conv blocks + AdaptiveAvgPool(8,8) + FC)",
        "head_desc": f"Linear(64*8*8, 128) → ReLU → Dropout(p={DROPOUT_P}) → Linear(128, 1)",
        "pretrained": False,
        "n_parameters": n_params,
        "dataset_name": "CIFAR-10 (torchvision)",
        "image_dir": str(CIFAR_DIR.relative_to(Path(PROJECT_ROOT))),
        "random_seed": RANDOM_SEED,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_positive_train": n_pos_train, "n_positive_val": n_pos_val,
        "aggregation": "1 image = 1 échantillon, label binaire one-vs-all",
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "img_size": img_size, "device": device,
        "warmup_epochs": warmup_iters, "patience": patience,
        "augmentation": "Resize → RandomCrop(pad=img_size/8) → HFlip",
        "loss": loss_desc,
        "sampler": "shuffle natif (pas de WeightedRandomSampler)",
        "num_workers": NUM_WORKERS,
        "git_commit": get_git_commit(Path(PROJECT_ROOT)),
        "started_at": started_at,
        "ended_at": None,
        "total_time_s": None,
        "total_time_human": None,
        "epochs_ran": None,
        "early_stopped": False,
        "best_auc": None,
        "best_epoch": None,
    }
    write_args_json(run_dir, hyperparams)
    write_run_readme(run_dir, hyperparams)
    logs = {"hyperparams": hyperparams, "epochs": []}

    best_auc = 0.0
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]",
                   unit="batch", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_time = time.time() - t0

        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val] ",
                                     unit="batch", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs).squeeze(1)
                val_loss += loss_fn(logits, labels).item()
                preds.extend(torch.sigmoid(logits).cpu().tolist())
                targets.extend(labels.cpu().tolist())

        auc = (
            roc_auc_score(targets, preds)
            if len(set(targets)) > 1
            else float("nan")
        )
        is_best = not np.isnan(auc) and auc > best_auc
        if is_best:
            best_auc = auc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auc": auc,
                    "val_preds": preds,
                    "val_targets": targets,
                },
                ckpt_path,
            )
        else:
            epochs_since_best += 1

        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss / len(train_loader), 6),
            "val_loss": round(val_loss / len(val_loader), 6),
            "auc": round(auc, 6) if not np.isnan(auc) else None,
            "time_s": round(train_time),
            "is_best": is_best,
        })
        logs_path.write_text(json.dumps(logs, indent=2))

        flag = " ← best" if is_best else ""
        print(
            f"[{epoch:3d}/{epochs}]  "
            f"train_loss={train_loss / len(train_loader):.4f}  "
            f"val_loss={val_loss / len(val_loader):.4f}  "
            f"auc={auc:.4f}  "
            f"time={train_time:.0f}s{flag}"
        )

        if epochs_since_best >= patience:
            print(f"\nEarly stopping : pas d'amélioration depuis {patience} epochs "
                  f"(best={best_auc:.4f} à epoch {best_epoch}).")
            hyperparams["early_stopped"] = True
            break

    total_time_s = time.time() - t_start
    hyperparams["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hyperparams["total_time_s"] = round(total_time_s, 1)
    hyperparams["total_time_human"] = format_duration(total_time_s)
    hyperparams["epochs_ran"] = len(logs["epochs"])
    hyperparams["best_auc"] = round(best_auc, 4) if best_auc else None
    hyperparams["best_epoch"] = best_epoch if best_epoch else None
    write_args_json(run_dir, hyperparams)
    write_run_readme(run_dir, hyperparams)
    logs_path.write_text(json.dumps(logs, indent=2))

    print(f"\nMeilleur AUC val : {best_auc:.4f}  (epoch {best_epoch})  →  {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_preds = ckpt["val_preds"]
    best_targets = ckpt["val_targets"]

    if len(set(best_targets)) >= 2:
        fpr, tpr, _ = roc_curve(best_targets, best_preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {best_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard (0.5)")
        ax.set_xlabel("Taux faux positifs")
        ax.set_ylabel("Taux vrais positifs")
        ax.set_title(f"ROC {target} (SmallCNN) — val set (best epoch : {ckpt['epoch']})")
        ax.legend()
        fig.savefig(roc_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Courbe ROC : {roc_path}")

    print(f"\nTous les artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne SmallCNN sur CIFAR-10 binaire (one-vs-all).")
    parser.add_argument("--positive-class", type=int, default=0,
                        help=f"Classe positive (0..9). Défaut: 0 ({CIFAR_CLASSES[0]}). "
                             f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CIFAR_CLASSES))}")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--loss", choices=["bce", "bce-pos-weight", "focal"],
                        default="bce-pos-weight",
                        help="Loss à utiliser (défaut: bce-pos-weight, adaptée au 10/90)")
    args = parser.parse_args()

    train(
        positive_class=args.positive_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        device=args.device,
        patience=args.patience,
        loss_name=args.loss,
    )
