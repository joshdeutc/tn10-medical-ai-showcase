"""
ResNet18 pré-entraîné ImageNet — fine-tuning complet, cancer (malignant), 1024px, H100.

Même harnais que train_smallcnn_cancer_1024.py (AMP, BCE, WeightedRandomSampler,
métadonnées de run) mais :
- Modèle = torchvision.resnet18(weights=IMAGENET1K_V1), tête remplacée par Dropout+Linear(512,1)
- Fine-tuning COMPLET (toutes les couches entraînables), LR=1e-5
- Entrée 3 canaux (Grayscale→3ch, requis par ResNet)

Usage:
    python -m fine_tuning.train_resnet_cancer_1024
    python -m fine_tuning.train_resnet_cancer_1024 --epochs 50 --batch-size 64 --img-size 1024
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from fine_tuning.config import (
    DEVICE,
    EXAM_LIST_PATH,
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

IMG_SIZE = 1024
BATCH_SIZE = 64
NUM_WORKERS_LOCAL = 16
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR = CHECKPOINT_DIR / "runs"
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-5          # fine-tuning complet depuis ImageNet
DROPOUT_P = 0.5
VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]

TARGET = "cancer"
DATASET_NAME = "RSNA Breast Cancer Detection (2022, Kaggle)"
NUM_EPOCHS = 50
IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "preprocess_image", "cropped_1024",
)


# ─── Construction des paires (image, label cancer) ───────────────────────────

def _make_entries(exam_list: list) -> list[tuple[str, int]]:
    """(chemin_image, label_malignant) — label partagé par toutes les vues de l'exam."""
    entries = []
    skipped_no_file = 0
    for exam in exam_list:
        label = int(exam["cancer_label"]["malignant"])
        for view in VIEWS:
            for rel_path in exam.get(view, []):
                p = os.path.join(IMAGE_DIR, rel_path + ".png")
                if not os.path.exists(p):
                    skipped_no_file += 1
                    continue
                entries.append((p, label))
    if skipped_no_file:
        print(f"⚠️  {skipped_no_file} images introuvables dans cropped_1024 (skippées)")
    return entries


# ─── Dataset (3 canaux pour ResNet) ──────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, entries, mean, std, augment=False, img_size=IMG_SIZE):
        self.entries = entries
        aug = (
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=10)]
            if augment else []
        )
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def compute_dataset_stats(entries, img_size, cache_path):
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())
        print(f"Stats chargées depuis {cache_path.name} : mean={stats['mean']}  std={stats['std']}")
        return stats["mean"], stats["std"]
    raw = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ch_sum = torch.zeros(3)
    ch_sq = torch.zeros(3)
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


def _make_sampler(entries):
    labels = [e[1] for e in entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0.0
    w_neg = 1.0 / n_neg if n_neg else 0.0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── Modèle : ResNet18 pré-entraîné, fine-tuning complet ─────────────────────

def build_model(device, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # Tête de classification binaire (toutes les couches restent entraînables).
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(model.fc.in_features, 1),
    )
    return model.to(device)


# ─── Boucle d'entraînement ───────────────────────────────────────────────────

def train(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    patience: int = EARLY_STOP_PATIENCE,
    pretrained: bool = True,
    resume_from: Path | None = None,
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    use_amp = (device == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    model_tag = f"resnet18_{'pretrained' if pretrained else 'scratch'}_{img_size}px"
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
    stats_path = CHECKPOINT_DIR / f"train_stats_{img_size}_cancer_resnet.json"
    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    train_exams, val_exams = load_and_split(EXAM_LIST_PATH)
    train_entries = _make_entries(train_exams)
    val_entries = _make_entries(val_exams)

    n_pos_train = sum(e[1] for e in train_entries)
    n_pos_val = sum(e[1] for e in val_entries)
    print(f"Run dir : {run_dir}")
    print(f"Train : {len(train_entries)} images  |  cancer={n_pos_train} ({100*n_pos_train/max(1,len(train_entries)):.1f}%)")
    print(f"Val   : {len(val_entries)} images  |  cancer={n_pos_val} ({100*n_pos_val/max(1,len(val_entries)):.1f}%)")
    print(f"Device: {device}  |  AMP={use_amp}  |  batch_size={batch_size}  |  img_size={img_size}×{img_size}  |  pretrained={pretrained}")

    mean, std = compute_dataset_stats(train_entries, img_size, stats_path)

    train_ds = ImageDataset(train_entries, mean=mean, std=std, augment=True, img_size=img_size)
    val_ds = ImageDataset(val_entries, mean=mean, std=std, augment=False, img_size=img_size)

    nw = min(NUM_WORKERS_LOCAL, max(NUM_WORKERS, NUM_WORKERS_LOCAL))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=_make_sampler(train_entries),
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    model = build_model(device, pretrained=pretrained)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet18 ({'ImageNet' if pretrained else 'scratch'}) : {n_params:,} paramètres entraînables ({n_params/1e6:.2f} M)")

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

    # BCE simple : le WeightedRandomSampler équilibre déjà les batchs.
    loss_fn = nn.BCEWithLogitsLoss()
    loss_desc = "BCEWithLogitsLoss"
    print(f"Loss : {loss_desc}")

    hyperparams = {
        "target": TARGET,
        "model_arch": "resnet18 (torchvision)",
        "head_desc": "Sequential(Dropout(0.5), Linear(512, 1))",
        "label_source": "data.pkl → cancer_label['malignant']",
        "pretrained": pretrained,
        "finetuning": "complet (toutes les couches)",
        "n_parameters": n_params,
        "dataset_name": DATASET_NAME,
        "image_dir": str(IMAGE_DIR),
        "img_size": img_size,
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_positive_train": n_pos_train, "n_positive_val": n_pos_val,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "device": device, "amp": use_amp,
        "warmup_epochs": warmup_iters, "patience": patience,
        "augmentation": "hflip + rotation(±10°)",
        "loss": loss_desc,
        "sampler": "WeightedRandomSampler (équilibre cancer/sain)",
        "num_workers": nw,
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
    ax.plot(fpr, tpr, color="seagreen", lw=2, label=f"AUC = {best_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard (0.5)")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"ROC ResNet18 pretrained cancer 1024px — val (epoch {ckpt['epoch']})")
    ax.legend()
    fig.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe ROC → {roc_path}")
    print(f"Artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 pré-entraîné — fine-tuning cancer, 1024px, H100.")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE)
    parser.add_argument("--device",       type=str,   default=DEVICE)
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE)
    parser.add_argument("--scratch",      action="store_true",
                        help="Part de poids aléatoires au lieu d'ImageNet (désactive le pré-entraînement)")
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
        pretrained=not args.scratch,
        resume_from=Path(args.resume) if args.resume else None,
    )
