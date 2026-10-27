"""
Entraînement SmallCNN sur les mammographies RSNA — cible "normalité".

Variante de `train_resnet_normalite.py` qui remplace ResNet18 (~11 M params)
par le SmallCNN du notebook `exemple_basics/intro_pytorch.qmd` (~70 k params).
Tout le reste est identique : même split, mêmes labels, mêmes augmentations,
même sampler équilibré → comparaison apples-to-apples sur la capacité du modèle.

Le SmallCNN d'origine est figé à des entrées 32×32 (sa `Linear(64*8*8, 128)`
hardcode la taille spatiale). On insère une `AdaptiveAvgPool2d((8, 8))` juste
avant la tête pour qu'il accepte n'importe quelle résolution — ici 512×512
comme le ResNet18, sans perte de détail médical.

Usage:
    python -m fine_tuning.train_smallcnn_normalite
    python -m fine_tuning.train_smallcnn_normalite --epochs 100 --batch-size 32
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
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
RUNS_DIR        = CHECKPOINT_DIR / "runs"
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS   = 5
LEARNING_RATE   = 1e-3   # LR du notebook (Adam, from-scratch)
DROPOUT_P       = 0.3    # dropout du SmallCNN d'origine (intro_pytorch.qmd)
VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
NORMALITE_CSV = Path(__file__).parent.parent / "data" / "rsna_images" / "train_subset.csv"

TARGET = "normalite"
DATASET_NAME = "RSNA Breast Cancer Detection (2022, Kaggle)"


# ─── Labels normalité : lecture du CSV ───────────────────────────────────────

def _load_normalite_labels() -> dict[tuple[int, int], int]:
    """Lit train_subset.csv et applique la règle de creation_column.py.

    Retourne {(patient_id, image_id): 0/1} où 1 = anormal.
    """
    if not NORMALITE_CSV.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {NORMALITE_CSV}\n"
            "Vérifie que data/raw/rsna_images/train_subset.csv est bien présent."
        )
    df = pd.read_csv(NORMALITE_CSV)
    anormal = (
        (df["cancer"] == 1)
        | (df["biopsy"] == 1)
        | (df["difficult_negative_case"] == True)
        | (df["BIRADS"] == 0)  # noqa: E712
    )
    df["label"] = anormal.astype(int)
    return {
        (int(r.patient_id), int(r.image_id)): int(r.label)
        for r in df.itertuples()
    }


# ─── Dataset image-level ─────────────────────────────────────────────────────

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
        print(f"⚠️  {skipped_no_label} images sans label dans le CSV (skippées)")
    return entries


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
        # Seul flip horizontal : un sein gauche flippé reste un sein anatomiquement valide
        # et le label normalité est latéralité-agnostique. RandomAffine et ColorJitter
        # avaient été ajoutés par défaut "parce que c'est courant" → retirés tant qu'on
        # n'a pas de preuve qu'ils aident sur ce dataset (à valider par tests préalables).
        aug = [transforms.RandomHorizontalFlip()] if augment else []
        # Mammographies = grayscale → 1 seul canal (vs 3 dans les scripts ResNet
        # qui doivent matcher l'attente RGB des poids ImageNet pré-entraînés).
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
    ch_sq  = torch.zeros(1)
    n_pix  = 0
    for path, _ in tqdm(entries, desc="Calcul mean/std train", unit="img"):
        t = raw(Image.open(path).convert("L"))
        ch_sum += t.sum(dim=[1, 2])
        ch_sq  += (t ** 2).sum(dim=[1, 2])
        n_pix  += t.shape[1] * t.shape[2]
    mean = (ch_sum / n_pix).tolist()
    std  = torch.sqrt(ch_sq / n_pix - (ch_sum / n_pix) ** 2).tolist()

    cache_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Stats train calculées : mean={mean}  std={std}  (sauvé → {cache_path.name})")
    return mean, std


def _make_sampler(entries: list) -> WeightedRandomSampler:
    labels = [e[1] for e in entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0.0
    w_neg = 1.0 / n_neg if n_neg else 0.0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── Modèle : SmallCNN du notebook intro_pytorch.qmd ─────────────────────────

class SmallCNN(nn.Module):
    """Reprise du SmallCNN du notebook, généralisé à n'importe quelle taille
    d'entrée via `AdaptiveAvgPool2d((8, 8))` avant la tête.

    Architecture :
        Conv(3→32) → BN → ReLU → MaxPool   (× 1)
        Conv(32→64) → BN → ReLU → MaxPool  (× 1)
        AdaptiveAvgPool((8, 8))            (← seul ajout vs notebook)
        Flatten → Linear(64*8*8, 128) → ReLU → Dropout → Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = 1, dropout: float = DROPOUT_P):
        super().__init__()
        # Bloc 1 : 1 → 32 canaux (mammographies = grayscale, pas RGB)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Bloc 2 : 32 → 64 canaux
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # Pool adaptatif → 8×8 quelle que soit la résolution d'entrée
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        # Tête (identique au notebook)
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
    return SmallCNN(num_classes=1, dropout=DROPOUT_P).to(device)


# ─── Boucle d'entraînement ───────────────────────────────────────────────────

class FocalLossBinary(nn.Module):
    """Focal loss binaire (Lin et al. 2017). alpha pondère la classe positive."""

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


def train(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    patience: int = EARLY_STOP_PATIENCE,
    resume_from: Path | None = None,
    loss_name: str = "bce",
    sampler_choice: str = "auto",
) -> None:
    if loss_name not in {"bce", "bce-pos-weight", "focal"}:
        raise ValueError(f"loss_name doit être 'bce', 'bce-pos-weight' ou 'focal' (reçu: {loss_name})")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model_tag = "smallcnn" if loss_name == "bce" else f"smallcnn-{loss_name}"
    if sampler_choice == "balanced" and loss_name != "bce":
        model_tag = f"{model_tag}-balanced"
    if resume_from is not None:
        run_dir = Path(resume_from)
        if not (run_dir / "last.pt").exists():
            raise FileNotFoundError(
                f"--resume : aucun last.pt dans {run_dir}\n"
                "Le run a peut-être été interrompu avant la fin du 1er epoch."
            )
        print(f"--resume : reprise depuis {run_dir}")
    else:
        run_dir = make_run_dir(RUNS_DIR, TARGET, model_tag)
    ckpt_path      = run_dir / "best.pt"
    ckpt_last_path = run_dir / "last.pt"
    logs_path = run_dir / "logs.json"
    roc_path  = run_dir / "roc.png"
    # Cache mean/std spécifique au pipeline grayscale 1 canal
    # (les scripts ResNet utilisent train_stats_{img_size}.json sur 3 canaux dupliqués)
    stats_path = CHECKPOINT_DIR / f"train_stats_{img_size}_grayscale.json"
    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    label_lookup = _load_normalite_labels()
    n_csv = len(label_lookup)
    n_pos_csv = sum(label_lookup.values())
    print(f"Labels normalité chargés : {n_csv} images  |  anormal={n_pos_csv} ({100*n_pos_csv/n_csv:.1f}%)")

    train_exams, val_exams = load_and_split(EXAM_LIST_PATH)

    train_entries = _make_entries(train_exams, label_lookup)
    val_entries   = _make_entries(val_exams,   label_lookup)

    n_pos_train = sum(e[1] for e in train_entries)
    n_pos_val   = sum(e[1] for e in val_entries)
    print(f"Run dir : {run_dir}")
    print(f"Train : {len(train_entries)} images  |  anormal={n_pos_train} "
          f"({100*n_pos_train/max(1,len(train_entries)):.1f}%)")
    print(f"Val   : {len(val_entries)} images  |  anormal={n_pos_val} "
          f"({100*n_pos_val/max(1,len(val_entries)):.1f}%)")
    print(f"Device: {device}  |  batch_size={batch_size}  |  img_size={img_size}x{img_size}")
    print(f"lr={lr}  |  weight_decay={weight_decay}  |  early-stop patience={patience}")
    print()

    mean, std = compute_dataset_stats(train_entries, img_size, stats_path)

    train_ds = ImageDataset(train_entries, mean=mean, std=std, augment=True,  img_size=img_size)
    val_ds   = ImageDataset(val_entries,   mean=mean, std=std, augment=False, img_size=img_size)

    if sampler_choice == "auto":
        use_sampler = (loss_name == "bce")
    elif sampler_choice == "balanced":
        use_sampler = True
    elif sampler_choice == "none":
        use_sampler = False
    else:
        raise ValueError(f"sampler doit être 'auto', 'balanced' ou 'none' (reçu: {sampler_choice})")

    if use_sampler:
        sampler_desc = "WeightedRandomSampler (équilibre normal / anormal)"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=_make_sampler(train_entries),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )
    else:
        sampler_desc = "aucun (distribution naturelle, loss pondérée)"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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
    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        loss_desc = "BCEWithLogitsLoss"
    elif loss_name == "bce-pos-weight":
        pos_weight = torch.tensor([n_pos_train and (len(train_entries) - n_pos_train) / n_pos_train or 1.0], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_desc = f"BCEWithLogitsLoss(pos_weight={pos_weight.item():.3f})"
    else:  # focal
        alpha = 1.0 - n_pos_train / max(1, len(train_entries))  # prior de la classe négative
        loss_fn = FocalLossBinary(alpha=alpha, gamma=2.0)
        loss_desc = f"FocalLossBinary(alpha={alpha:.3f}, gamma=2.0)"
    print(f"Loss : {loss_desc}")
    print(f"Sampler : {sampler_desc}")

    hyperparams = {
        "target": TARGET,
        "label_rule": "cancer==1 OR biopsy==1 OR difficult_negative_case==True OR BIRADS==0",
        "label_csv": str(NORMALITE_CSV.relative_to(NORMALITE_CSV.parents[2])),
        "model_arch": "SmallCNN (2 conv blocks + AdaptiveAvgPool(8,8) + FC) — du notebook intro_pytorch.qmd",
        "head_desc": f"Linear(64*8*8, 128) → ReLU → Dropout(p={DROPOUT_P}) → Linear(128, 1)",
        "pretrained": False,
        "n_parameters": n_params,
        "dataset_name": DATASET_NAME,
        "image_dir": str(IMAGE_DIR),
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_positive_train": n_pos_train, "n_positive_val": n_pos_val,
        "aggregation": "par image (chaque vue = un échantillon), label lu dans le CSV",
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "img_size": img_size, "device": device,
        "warmup_epochs": warmup_iters, "patience": patience,
        "augmentation": "hflip uniquement",
        "loss": loss_desc,
        "sampler": sampler_desc,
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
    # En reprise, on garde l'args.json original (started_at d'origine, etc.)
    # et on recharge logs.json pour continuer à appender.
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
        start_epoch        = state["start_epoch"]
        best_auc           = state["best_auc"]
        best_epoch         = state["best_epoch"]
        epochs_since_best  = state["epochs_since_best"]
        print(f"--resume : reprise à l'epoch {start_epoch}/{epochs}  "
              f"(best_auc={best_auc:.4f} à epoch {best_epoch})")

    for epoch in range(start_epoch, epochs + 1):
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

        # Snapshot de reprise — sauvé après CHAQUE epoch (best ou pas).
        # Permet à `--resume <run_dir>` de repartir à l'epoch+1 si crash/reboot.
        save_last_checkpoint(
            ckpt_last_path, epoch, model, optimizer, scheduler,
            best_auc, best_epoch, epochs_since_best,
        )

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
    best_preds   = ckpt["val_preds"]
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
    ax.set_title(f"Courbe ROC normalité (SmallCNN) — val set (meilleur epoch : {ckpt['epoch']})")
    ax.legend()
    fig.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe ROC sauvegardée → {roc_path}")
    print(f"\nTous les artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne SmallCNN sur le label normalité (RSNA).")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS,    help=f"Nombre d'epochs (défaut: {NUM_EPOCHS})")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,    help=f"Taille de batch (défaut: {BATCH_SIZE})")
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE, help=f"Learning rate (défaut: {LEARNING_RATE})")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,  help=f"L2 regularisation (défaut: {WEIGHT_DECAY})")
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE,      help=f"Taille des images carrées (défaut: {IMG_SIZE})")
    parser.add_argument("--device",       type=str,   default=DEVICE,        help=f"Device pytorch (défaut: {DEVICE})")
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE, help=f"Epochs sans amélioration avant early stop (défaut: {EARLY_STOP_PATIENCE})")
    parser.add_argument("--resume",       type=str,   default=None,          help="Chemin d'un run interrompu à reprendre (ex: fine_tuning/checkpoints/runs/normalite/smallcnn/20260427-162128)")
    parser.add_argument("--loss",         type=str,   default="bce",
                        choices=["bce", "bce-pos-weight", "focal"],
                        help="bce = BCE classique | "
                             "bce-pos-weight = BCE avec pos_weight automatique | "
                             "focal = focal loss (alpha auto, gamma=2)")
    parser.add_argument("--sampler",      type=str,   default="auto",
                        choices=["auto", "balanced", "none"],
                        help="auto = sampler équilibré si bce sinon aucun (défaut, historique) | "
                             "balanced = WeightedRandomSampler quelle que soit la loss | "
                             "none = pas de sampler, shuffle natif")
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
        sampler_choice=args.sampler,
    )
