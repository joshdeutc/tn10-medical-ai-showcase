"""
SmallCNN v2 — entraînement cancer du sein (normalité), optimisé L40S.

Améliorations vs train_smallcnn_normalite.py :
- Images 1024×1024 depuis cropped_images/ (vs 512×512 cropped_512/)
- SmallCNN v2 : 3 blocs conv (1→32→64→128), AdaptiveAvgPool(16×16)
- Mixed precision AMP (2-3× plus rapide, moitié moins de VRAM)
- batch_size=64 (vs 24)
- Augmentation enrichie : flip H+V + rotation ±10°
- 100 epochs, patience=15

Usage:
    python -m fine_tuning.train_smallcnn_cancer_v2
    python -m fine_tuning.train_smallcnn_cancer_v2 --epochs 100 --batch-size 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from fine_tuning.config import (
    DEVICE,
    EXAM_LIST_PATH,
    IMAGE_DIR_LARGE,
    NUM_WORKERS,
    PROJECT_ROOT,
    RANDOM_SEED,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from fine_tuning.dataset import load_and_split
from fine_tuning.run_metadata import (
    format_duration,
    get_git_commit,
    load_last_checkpoint,
    make_run_dir,
    save_last_checkpoint,
    write_args_json,
    write_run_readme,
)

torch.backends.cudnn.benchmark = True

# ─── Hyperparamètres ─────────────────────────────────────────────────────────

IMG_SIZE = 512
BATCH_SIZE = 256
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR = CHECKPOINT_DIR / "runs"
EARLY_STOP_PATIENCE = 15
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-3
DROPOUT_P = 0.3
VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
NORMALITE_CSV = Path(__file__).parent.parent / "data" / "rsna_images" / "train.csv"

TARGET = "normalite"
DATASET_NAME = "RSNA Breast Cancer Detection (2022, Kaggle)"
NUM_EPOCHS = 100
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preprocess_image", "cropped_512")


# ─── Labels normalité ────────────────────────────────────────────────────────

def _load_normalite_labels() -> dict[tuple[int, int], int]:
    if not NORMALITE_CSV.exists():
        raise FileNotFoundError(f"CSV introuvable : {NORMALITE_CSV}")
    df = pd.read_csv(NORMALITE_CSV)
    anormal = (
        (df["cancer"] == 1)
        | (df["biopsy"] == 1)
        | (df["difficult_negative_case"] == True)
        | (df["BIRADS"] == 0)
    )
    df["label"] = anormal.astype(int)
    return {
        (int(r.patient_id), int(r.image_id)): int(r.label)
        for r in df.itertuples()
    }


def _make_entries(
    exam_list: list,
    label_lookup: dict[tuple[int, int], int],
) -> list[tuple[str, int]]:
    entries = []
    skipped_no_file = 0
    skipped_no_label = 0
    for exam in exam_list:
        for view in VIEWS:
            for rel_path in exam.get(view, []):
                p = os.path.join(IMAGE_DIR, rel_path + ".png")
                if not os.path.exists(p):
                    skipped_no_file += 1
                    continue
                pid_str, iid_str = rel_path.split("/")
                key = (int(pid_str), int(iid_str))
                if key not in label_lookup:
                    skipped_no_label += 1
                    continue
                entries.append((p, label_lookup[key]))
    if skipped_no_file:
        print(f"⚠️  {skipped_no_file} images introuvables (skippées)")
    if skipped_no_label:
        print(f"⚠️  {skipped_no_label} images sans label dans le CSV (skippées)")
    return entries


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[str, int]],
        mean: list[float],
        std: list[float],
        augment: bool = False,
        img_size: int = IMG_SIZE,
    ):
        self.entries = entries
        aug = (
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
            ]
            if augment
            else []
        )
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                *aug,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def compute_dataset_stats(
    entries: list[tuple[str, int]],
    img_size: int,
    cache_path: Path,
) -> tuple[list[float], list[float]]:
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())
        print(f"Stats chargées depuis {cache_path.name} : mean={stats['mean']}  std={stats['std']}")
        return stats["mean"], stats["std"]

    raw = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ch_sum = torch.zeros(1)
    ch_sq = torch.zeros(1)
    n_pix = 0
    for path, _ in tqdm(entries, desc="Calcul mean/std train", unit="img"):
        t = raw(Image.open(path).convert("L"))
        ch_sum += t.sum(dim=[1, 2])
        ch_sq += (t ** 2).sum(dim=[1, 2])
        n_pix += t.shape[1] * t.shape[2]
    mean = (ch_sum / n_pix).tolist()
    std = torch.sqrt(ch_sq / n_pix - (ch_sum / n_pix) ** 2).tolist()
    cache_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Stats calculées : mean={mean}  std={std}  → {cache_path.name}")
    return mean, std


def _make_sampler(entries: list) -> WeightedRandomSampler:
    labels = [e[1] for e in entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0.0
    w_neg = 1.0 / n_neg if n_neg else 0.0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── Modèle : SmallCNN v2 ────────────────────────────────────────────────────

class SmallCNNv2(nn.Module):
    """SmallCNN v2 — 3 blocs conv + AdaptiveAvgPool(16×16).

    Architecture :
        Conv(1→32)  → BN → ReLU → MaxPool(2)   # 1024 → 512
        Conv(32→64) → BN → ReLU → MaxPool(2)   # 512  → 256
        Conv(64→128)→ BN → ReLU → MaxPool(2)   # 256  → 128
        AdaptiveAvgPool((16, 16))               # → 128×16×16 = 32768
        Flatten
        Linear(32768, 512) → ReLU → Dropout
        Linear(512, 1)
    """

    def __init__(self, dropout: float = DROPOUT_P):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_model(device: str) -> nn.Module:
    return SmallCNNv2(dropout=DROPOUT_P).to(device)


# ─── Loss ────────────────────────────────────────────────────────────────────

class FocalLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.84, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ─── Boucle d'entraînement ───────────────────────────────────────────────────

def train(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    patience: int = EARLY_STOP_PATIENCE,
    resume_from: Path | None = None,
    loss_name: str = "focal",
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    use_amp = (device == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    model_tag = f"smallcnn-v2-{loss_name}-{img_size}px"
    if resume_from is not None:
        run_dir = Path(resume_from)
        if not (run_dir / "last.pt").exists():
            raise FileNotFoundError(f"--resume : aucun last.pt dans {run_dir}")
        print(f"--resume : reprise depuis {run_dir}")
    else:
        run_dir = make_run_dir(RUNS_DIR, TARGET, model_tag)

    ckpt_path = run_dir / "best.pt"
    ckpt_last_path = run_dir / "last.pt"
    logs_path = run_dir / "logs.json"
    roc_path = run_dir / "roc.png"
    stats_path = CHECKPOINT_DIR / f"train_stats_{img_size}_grayscale_large.json"
    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    label_lookup = _load_normalite_labels()
    n_csv = len(label_lookup)
    n_pos_csv = sum(label_lookup.values())
    print(f"Labels : {n_csv} images  |  anormal={n_pos_csv} ({100*n_pos_csv/n_csv:.1f}%)")

    train_exams, val_exams = load_and_split(EXAM_LIST_PATH)
    train_entries = _make_entries(train_exams, label_lookup)
    val_entries = _make_entries(val_exams, label_lookup)

    n_pos_train = sum(e[1] for e in train_entries)
    n_pos_val = sum(e[1] for e in val_entries)
    print(f"Run dir : {run_dir}")
    print(f"Train : {len(train_entries)} images  |  anormal={n_pos_train} ({100*n_pos_train/max(1,len(train_entries)):.1f}%)")
    print(f"Val   : {len(val_entries)} images  |  anormal={n_pos_val} ({100*n_pos_val/max(1,len(val_entries)):.1f}%)")
    print(f"Device: {device}  |  AMP={use_amp}  |  batch_size={batch_size}  |  img_size={img_size}×{img_size}")
    print(f"lr={lr}  |  weight_decay={weight_decay}  |  patience={patience}\n")

    mean, std = compute_dataset_stats(train_entries, img_size, stats_path)

    train_ds = ImageDataset(train_entries, mean=mean, std=std, augment=True, img_size=img_size)
    val_ds = ImageDataset(val_entries, mean=mean, std=std, augment=False, img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=_make_sampler(train_entries),
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SmallCNNv2 : {n_params:,} paramètres ({n_params/1e6:.2f} M)")

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

    if loss_name == "focal":
        alpha = 1.0 - n_pos_train / max(1, len(train_entries))
        loss_fn = FocalLossBinary(alpha=alpha, gamma=2.0)
        loss_desc = f"FocalLossBinary(alpha={alpha:.3f}, gamma=2.0)"
    elif loss_name == "bce-pos-weight":
        pos_weight = torch.tensor(
            [(len(train_entries) - n_pos_train) / max(1, n_pos_train)],
            device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_desc = f"BCEWithLogitsLoss(pos_weight={pos_weight.item():.3f})"
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        loss_desc = "BCEWithLogitsLoss"
    print(f"Loss : {loss_desc}")

    hyperparams = {
        "target": TARGET,
        "model_arch": "SmallCNNv2 (3 conv blocks + AdaptiveAvgPool(16,16) + FC512)",
        "pretrained": False,
        "n_parameters": n_params,
        "dataset_name": DATASET_NAME,
        "image_dir": str(IMAGE_DIR),
        "img_size": img_size,
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_positive_train": n_pos_train, "n_positive_val": n_pos_val,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "device": device,
        "amp": use_amp,
        "warmup_epochs": warmup_iters, "patience": patience,
        "augmentation": "hflip + vflip(p=0.2) + rotation(±10°)",
        "loss": loss_desc,
        "sampler": "WeightedRandomSampler (équilibre normal/anormal)",
        "num_workers": NUM_WORKERS,
        "git_commit": get_git_commit(Path(PROJECT_ROOT)),
        "started_at": started_at,
        "ended_at": None, "total_time_s": None, "total_time_human": None,
        "epochs_ran": None, "early_stopped": False,
        "best_auc": None, "best_epoch": None,
    }

    if resume_from is None:
        write_args_json(run_dir, hyperparams)
        write_run_readme(run_dir, hyperparams)
        logs = {"hyperparams": hyperparams, "epochs": []}
    else:
        logs = json.loads(logs_path.read_text()) if logs_path.exists() else {"hyperparams": hyperparams, "epochs": []}

    best_auc = 0.0
    best_epoch = 0
    epochs_since_best = 0
    start_epoch = 1

    if resume_from is not None:
        state = load_last_checkpoint(ckpt_last_path, model, optimizer, scheduler, device)
        start_epoch = state["start_epoch"]
        best_auc = state["best_auc"]
        best_epoch = state["best_epoch"]
        epochs_since_best = state["epochs_since_best"]
        print(f"--resume : epoch {start_epoch}/{epochs}  (best_auc={best_auc:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]", unit="batch", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(imgs).squeeze(1)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_time = time.time() - t0

        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val] ", unit="batch", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast("cuda", enabled=use_amp):
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

        save_last_checkpoint(
            ckpt_last_path, epoch, model, optimizer, scheduler,
            best_auc, best_epoch, epochs_since_best,
        )

        flag = " ← best" if is_best else ""
        print(
            f"[{epoch:3d}/{epochs}]  "
            f"train_loss={train_loss/len(train_loader):.4f}  "
            f"val_loss={val_loss/len(val_loader):.4f}  "
            f"auc={auc:.4f}  "
            f"time={train_time:.0f}s{flag}"
        )

        if epochs_since_best >= patience:
            print(f"\nEarly stopping : pas d'amélioration depuis {patience} epochs "
                  f"(best={best_auc:.4f} @ epoch {best_epoch}).")
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

    if len(set(best_targets)) < 2:
        print("Pas assez de classes distinctes pour tracer la courbe ROC.")
        return

    fpr, tpr, _ = roc_curve(best_targets, best_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {best_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard (0.5)")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"ROC SmallCNNv2 — val (meilleur epoch : {ckpt['epoch']})")
    ax.legend()
    fig.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe ROC → {roc_path}")
    print(f"Artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmallCNN v2 — cancer du sein (normalité), optimisé L40S.")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE)
    parser.add_argument("--device",       type=str,   default=DEVICE)
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--loss",         type=str,   default="focal",
                        choices=["bce", "bce-pos-weight", "focal"])
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        device=args.device,
        patience=args.patience,
        resume_from=Path(args.resume) if args.resume else None,
        loss_name=args.loss,
    )
