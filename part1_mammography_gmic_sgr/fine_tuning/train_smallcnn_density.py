"""
Entraînement SmallCNN sur RSNA — cible "density" (multi-classe BIRADS A/B/C/D).

Variante multi-classe de `train_smallcnn_normalite.py` :
- 4 sorties au lieu de 1
- CrossEntropyLoss avec poids de classe (déséquilibre ~9× entre C et D)
- Métrique principale : macro-F1 (insensible au déséquilibre)
- Visualisation : matrice de confusion au lieu de ROC
- Entrée grayscale 1 canal (contrairement à ResNet qui duplique sur 3 canaux)

Usage:
    python -m fine_tuning.train_smallcnn_density
    python -m fine_tuning.train_smallcnn_density --epochs 100 --batch-size 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from fine_tuning.config import (
    DEVICE,
    EXAM_LIST_PATH,
    IMAGE_DIR,
    NUM_EPOCHS,
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
BATCH_SIZE = 24
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR = CHECKPOINT_DIR / "runs"
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-3   # LR from-scratch (identique à train_smallcnn_normalite)
DROPOUT_P = 0.3
VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
DENSITY_CSV = Path(__file__).parent.parent / "data" / "rsna_images" / "train.csv"

TARGET = "density"
DATASET_NAME = "RSNA Breast Cancer Detection (2022, Kaggle)"
CLASSES = ["A", "B", "C", "D"]
NUM_CLASSES = 4
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}


# ─── Labels density : lecture du CSV ─────────────────────────────────────────

def _load_density_labels() -> dict[tuple[int, int], int]:
    """{(patient_id, image_id): 0/1/2/3} où 0=A, 1=B, 2=C, 3=D.
    Les images sans density (NaN) sont exclues du dictionnaire."""
    if not DENSITY_CSV.exists():
        raise FileNotFoundError(f"CSV introuvable : {DENSITY_CSV}")
    df = pd.read_csv(DENSITY_CSV).dropna(subset=["density"])
    return {
        (int(r.patient_id), int(r.image_id)): LABEL_MAP[r.density]
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
    if skipped_no_label:
        print(f"⚠️  {skipped_no_label} images sans density connue (skippées)")
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
        aug = [transforms.RandomHorizontalFlip()] if augment else []
        # Grayscale 1 canal : la densité mammaire est une propriété de texture,
        # pas de couleur. Évite de dupliquer inutilement sur 3 canaux.
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
        return self.transform(img), torch.tensor(label, dtype=torch.long)


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
    print(f"Stats train : mean={mean}  std={std}  (sauvé → {cache_path.name})")
    return mean, std


def _make_sampler(entries: list) -> WeightedRandomSampler:
    labels = np.array([e[1] for e in entries])
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = class_weights[labels].tolist()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _compute_class_weights(entries: list, device: str) -> torch.Tensor:
    labels = np.array([e[1] for e in entries])
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights *= NUM_CLASSES / weights.sum()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─── Modèle : SmallCNN du notebook intro_pytorch.qmd ─────────────────────────

class SmallCNN(nn.Module):
    """SmallCNN adapté à n'importe quelle résolution via AdaptiveAvgPool2d((8,8)).

    Architecture :
        Conv(1→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → MaxPool
        AdaptiveAvgPool((8, 8))
        Flatten → Linear(64*8*8, 128) → ReLU → Dropout → Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_P):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
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


def build_smallcnn(device: str) -> nn.Module:
    return SmallCNN(num_classes=NUM_CLASSES, dropout=DROPOUT_P).to(device)


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
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model_tag = "smallcnn"
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
    cm_path = run_dir / "confusion_matrix.png"
    # Réutilise le cache grayscale existant (partagé avec train_smallcnn_normalite)
    stats_path = CHECKPOINT_DIR / f"train_stats_{img_size}_grayscale.json"
    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    label_lookup = _load_density_labels()
    n_csv = len(label_lookup)
    print(f"Labels density : {n_csv} images réparties sur {NUM_CLASSES} classes")
    counts_csv = np.bincount(list(label_lookup.values()), minlength=NUM_CLASSES)
    for cls, n in zip(CLASSES, counts_csv):
        print(f"  classe {cls} : {n} ({100*n/n_csv:.1f}%)")

    train_exams, val_exams = load_and_split(EXAM_LIST_PATH)
    train_entries = _make_entries(train_exams, label_lookup)
    val_entries = _make_entries(val_exams, label_lookup)

    train_counts = np.bincount([e[1] for e in train_entries], minlength=NUM_CLASSES)
    val_counts = np.bincount([e[1] for e in val_entries], minlength=NUM_CLASSES)
    print(f"\nRun dir : {run_dir}")
    print(f"Train : {len(train_entries)} images  |  {dict(zip(CLASSES, train_counts.tolist()))}")
    print(f"Val   : {len(val_entries)} images  |  {dict(zip(CLASSES, val_counts.tolist()))}")
    print(f"Device: {device}  |  batch_size={batch_size}  |  img_size={img_size}")
    print(f"lr={lr}  |  weight_decay={weight_decay}  |  patience={patience}\n")

    mean, std = compute_dataset_stats(train_entries, img_size, stats_path)

    train_ds = ImageDataset(train_entries, mean=mean, std=std, augment=True, img_size=img_size)
    val_ds = ImageDataset(val_entries, mean=mean, std=std, augment=False, img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=_make_sampler(train_entries),
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

    model = build_smallcnn(device)
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

    class_weights = _compute_class_weights(train_entries, device)
    print(f"Class weights (CE loss) : {class_weights.cpu().tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    hyperparams = {
        "target": TARGET,
        "label_rule": "density (A/B/C/D), images sans density exclues",
        "label_csv": str(DENSITY_CSV.relative_to(DENSITY_CSV.parents[2])),
        "num_classes": NUM_CLASSES,
        "model_arch": "SmallCNN (2 conv blocks + AdaptiveAvgPool(8,8) + FC) — du notebook intro_pytorch.qmd",
        "head_desc": f"Linear(64*8*8, 128) → ReLU → Dropout(p={DROPOUT_P}) → Linear(128, {NUM_CLASSES})",
        "pretrained": False,
        "n_parameters": n_params,
        "dataset_name": DATASET_NAME,
        "image_dir": str(IMAGE_DIR),
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_per_class_train": dict(zip(CLASSES, train_counts.tolist())),
        "n_per_class_val": dict(zip(CLASSES, val_counts.tolist())),
        "aggregation": "par image, label density lu dans le CSV",
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "img_size": img_size, "device": device,
        "warmup_epochs": warmup_iters, "patience": patience,
        "augmentation": "hflip uniquement",
        "sampler": "WeightedRandomSampler (équilibrage 4 classes)",
        "loss": "CrossEntropyLoss (poids inverses des fréquences)",
        "num_workers": NUM_WORKERS,
        "git_commit": get_git_commit(Path(PROJECT_ROOT)),
        "started_at": started_at,
        "ended_at": None, "total_time_s": None, "total_time_human": None,
        "epochs_ran": None, "early_stopped": False,
        "best_macro_f1": None, "best_epoch": None,
    }
    if resume_from is None:
        write_args_json(run_dir, hyperparams)
        write_run_readme(run_dir, hyperparams)
        logs = {"hyperparams": hyperparams, "epochs": []}
    else:
        logs = json.loads(logs_path.read_text()) if logs_path.exists() else {"hyperparams": hyperparams, "epochs": []}

    best_f1 = 0.0
    best_epoch = 0
    epochs_since_best = 0
    start_epoch = 1
    if resume_from is not None:
        state = load_last_checkpoint(ckpt_last_path, model, optimizer, scheduler, device)
        start_epoch = state["start_epoch"]
        best_f1 = state["best_auc"]  # champ réutilisé pour macro-F1
        best_epoch = state["best_epoch"]
        epochs_since_best = state["epochs_since_best"]
        print(f"--resume : epoch {start_epoch}/{epochs}  (best_f1={best_f1:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]", unit="batch", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_time = time.time() - t0

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val] ", unit="batch", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += loss_fn(logits, labels).item()
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))

        is_best = macro_f1 > best_f1
        if is_best:
            best_f1 = macro_f1
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": macro_f1,
                    "val_accuracy": accuracy,
                    "val_preds": all_preds,
                    "val_targets": all_targets,
                },
                ckpt_path,
            )
        else:
            epochs_since_best += 1

        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss / len(train_loader), 6),
            "val_loss": round(val_loss / len(val_loader), 6),
            "macro_f1": round(macro_f1, 6),
            "accuracy": round(accuracy, 6),
            "time_s": round(train_time),
            "is_best": is_best,
        })
        logs_path.write_text(json.dumps(logs, indent=2))

        save_last_checkpoint(
            ckpt_last_path, epoch, model, optimizer, scheduler,
            best_f1, best_epoch, epochs_since_best,
        )

        flag = " ← best" if is_best else ""
        print(
            f"[{epoch:3d}/{epochs}]  "
            f"train_loss={train_loss/len(train_loader):.4f}  "
            f"val_loss={val_loss/len(val_loader):.4f}  "
            f"macro_f1={macro_f1:.4f}  "
            f"acc={accuracy:.4f}  "
            f"time={train_time:.0f}s{flag}"
        )

        if epochs_since_best >= patience:
            print(f"\nEarly stopping : pas d'amélioration depuis {patience} epochs "
                  f"(best={best_f1:.4f} @ epoch {best_epoch}).")
            hyperparams["early_stopped"] = True
            break

    total_time_s = time.time() - t_start
    hyperparams["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hyperparams["total_time_s"] = round(total_time_s, 1)
    hyperparams["total_time_human"] = format_duration(total_time_s)
    hyperparams["epochs_ran"] = len(logs["epochs"])
    hyperparams["best_macro_f1"] = round(best_f1, 4) if best_f1 else None
    hyperparams["best_epoch"] = best_epoch if best_epoch else None
    write_args_json(run_dir, hyperparams)
    write_run_readme(run_dir, hyperparams)
    logs_path.write_text(json.dumps(logs, indent=2))

    print(f"\nMeilleur macro-F1 val : {best_f1:.4f}  (epoch {best_epoch})  →  {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cm = confusion_matrix(ckpt["val_targets"], ckpt["val_preds"], labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité")
    ax.set_title(f"Matrice de confusion val — epoch {ckpt['epoch']} (macro-F1={best_f1:.3f})")
    fig.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Matrice de confusion sauvegardée → {cm_path}")
    print(f"\nTous les artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne SmallCNN multi-classe sur la densité (A/B/C/D).")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE)
    parser.add_argument("--device",       type=str,   default=DEVICE)
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE)
    parser.add_argument("--resume",       type=str,   default=None)
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
    )
