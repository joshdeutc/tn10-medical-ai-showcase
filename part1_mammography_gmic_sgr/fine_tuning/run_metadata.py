"""
Métadonnées et organisation des runs d'entraînement.

Pour chaque entraînement, on produit :
- une arborescence `runs/{target}/{model_tag}/{timestamp}/`
- un `args.json` enrichi (source de vérité machine)
- un `README.md` humain, mis à jour à la fin du run pour inclure les résultats

Cette logique est partagée entre `train_resnet.py` (cible cancer) et
`train_resnet_normalite.py` (cible normalité).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

# ─── Catalogue des cibles ────────────────────────────────────────────────────
# On décrit chaque target une fois ici ; les scripts train y pointent via
# la clé (`"cancer_malignant"`, `"normalite"`, …). Ajouter une nouvelle cible
# = ajouter une entrée dans ce dict.

TARGET_CATALOG: dict[str, dict[str, str]] = {
    "cancer_malignant": {
        "column": "cancer",
        "definition": (
            "1 si une tumeur maligne a été confirmée (biopsie/chirurgie), "
            "0 sinon. Label par sein dans le CSV RSNA, agrégé par image "
            "dans ce run (chaque vue = un échantillon)."
        ),
    },
    "normalite": {
        "column": "dérivé (cancer | biopsy | difficult_negative_case)",
        "definition": (
            "1 si l'une des colonnes est vraie pour l'image : "
            "cancer==1 OR biopsy==1 OR difficult_negative_case==True. "
            "Plus permissif que `cancer` seul — utile quand les positifs "
            "cancer sont trop rares pour entraîner directement."
        ),
    },
    "density": {
        "column": "density",
        "definition": (
            "Catégorie BIRADS de densité mammaire (A/B/C/D). Classification "
            "multi-classe à 4 sorties. Images sans density renseignée exclues."
        ),
    },
    "cancer_malignant_gmic": {
        "column": "cancer (+ biopsy pour le canal benign)",
        "definition": (
            "Sorties GMIC 2 canaux [benign, malignant] par sein : "
            "malignant = cancer==1 ; benign = biopsy==1 & cancer==0. "
            "Entraînement par VUE (GMIC traite une vue à la fois), mais "
            "sélection/early-stopping sur l'AUC malignant agrégée par SEIN "
            "(patient_id+latéralité, moyenne des probas CC+MLO)."
        ),
    },
}


# ─── Arborescence ────────────────────────────────────────────────────────────

def make_run_dir(
    runs_root: Path,
    target: str,
    model_tag: str,
    timestamp: str | None = None,
) -> Path:
    """
    Crée `runs_root/{target}/{model_tag}/{timestamp}/` et le retourne.

    - `target`     : clé de TARGET_CATALOG (ex: "cancer_malignant")
    - `model_tag`  : libellé modèle+init (ex: "resnet18_pretrained")
    - `timestamp`  : YYYYMMDD-HHMMSS, auto si None
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / target / model_tag / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ─── Git commit ──────────────────────────────────────────────────────────────

def get_git_commit(project_root: Path) -> str:
    """Retourne le hash court du HEAD, ou 'unknown' en cas d'erreur."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        commit = out.decode().strip()
        # Ajoute un suffixe -dirty si l'arbre de travail a des modifs non committées
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if status:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


# ─── Écriture du README humain ───────────────────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None:
        return "_(pas encore)_"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def write_run_readme(run_dir: Path, meta: dict[str, Any]) -> None:
    """
    Écrit `run_dir/README.md` à partir de la dict `meta`.

    Appelable deux fois :
    - au début du run avec les champs résultats à None
    - à la fin avec tous les champs remplis (remplace le README initial)
    """
    target = meta["target"]
    target_info = TARGET_CATALOG.get(target, {"column": "?", "definition": "?"})

    lines = [
        f"# Run {run_dir.name}",
        "",
        "## Cible",
        f"- **Nom interne** : `{target}`",
        f"- **Colonne source** : `{target_info['column']}`",
        f"- **Définition** : {target_info['definition']}",
        f"- **Agrégation** : {meta.get('aggregation', 'par image (chaque vue = un échantillon)')}",
        "",
        "## Modèle",
        f"- **Architecture** : {meta.get('model_arch', '?')}",
        # init_desc : description libre prioritaire (ex: poids NYU pour GMIC)
        f"- **Initialisation** : "
        + (meta["init_desc"] if meta.get("init_desc") else
           ("ImageNet1K_V1 (fine-tuning)" if meta.get("pretrained") else "aléatoire (from-scratch)")),
        f"- **Tête de classification** : {meta.get('head_desc', '?')}",
        "",
        "## Données",
        f"- **Dataset** : {meta.get('dataset_name', '?')}",
        f"- **Dossier images** : `{meta.get('image_dir', '?')}`",
        # img_size : int -> carré "S×S" ; str -> tel quel (ex: "2944×1920 (H×W)")
        f"- **Taille images (modèle)** : "
        + (f"{meta.get('img_size')}×{meta.get('img_size')}"
           if isinstance(meta.get("img_size"), int) else str(meta.get("img_size", "?"))),
        # split_desc : description libre prioritaire sur le format val_split standard
        f"- **Split** : "
        + (meta["split_desc"] if meta.get("split_desc") else
           f"{100*(1-meta.get('val_split',0)):.0f} % train / {100*meta.get('val_split',0):.0f} % val, stratifié au niveau exam, seed={meta.get('random_seed','?')}"),
        f"- **Train** : {meta.get('n_train','?')} images / {meta.get('n_positive_train','?')} positifs",
        f"- **Val**   : {meta.get('n_val','?')} images / {meta.get('n_positive_val','?')} positifs",
        "",
        "## Entraînement",
        f"- **Date début** : {meta.get('started_at', '?')}",
        f"- **Date fin**   : {_fmt(meta.get('ended_at'))}",
        f"- **Device** : {meta.get('device','?')}",
        f"- **Batch size** : {meta.get('batch_size','?')}  |  Epochs max : {meta.get('epochs','?')}  |  Early stop patience : {meta.get('patience','?')}",
        f"- **Optimiseur** : Adam, lr={meta.get('lr','?')}, weight_decay={meta.get('weight_decay','?')}",
        f"- **Scheduler** : warmup {meta.get('warmup_epochs','?')} epochs → cosine annealing",
        f"- **Augmentation** : {meta.get('augmentation','?')}",
        f"- **Sampler** : {meta.get('sampler','?')}",
        f"- **Git commit** : `{meta.get('git_commit','unknown')}`",
        "",
        "## Résultats",
        f"- **Best AUC val** : {_fmt(meta.get('best_auc'))}" + (f" (epoch {meta['best_epoch']})" if meta.get('best_epoch') else ""),
        f"- **Temps total** : {_fmt(meta.get('total_time_human'))}",
        f"- **Epochs exécutées** : {_fmt(meta.get('epochs_ran'))} / {meta.get('epochs','?')}"
        + (" (early-stopped)" if meta.get("early_stopped") else ""),
        "",
        "## Fichiers",
    ] + (meta["files_desc"] if meta.get("files_desc") else [
        "- `best.pt`    — poids du meilleur epoch (state_dict + val_preds/val_targets)",
        "- `logs.json`  — métriques par epoch",
        "- `args.json`  — hyperparams bruts (source de vérité machine)",
        "- `roc.png`    — courbe ROC du meilleur epoch (val set)",
    ]) + [
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines))


# ─── Persistance args.json ───────────────────────────────────────────────────

def write_args_json(run_dir: Path, meta: dict[str, Any]) -> None:
    """Dump `meta` en JSON dans `run_dir/args.json` (source de vérité machine)."""
    (run_dir / "args.json").write_text(json.dumps(meta, indent=2, default=str))


# ─── Checkpoint reprise (--resume) ───────────────────────────────────────────
# `last.pt` contient tout l'état nécessaire pour reprendre un run interrompu :
# - poids modèle
# - état Adam (momentum, etc.)
# - état du scheduler (warmup → cosine)
# - bookkeeping early-stop (best_auc / best_epoch / epochs_since_best)
# Sauvé après chaque epoch — un crash perd au pire l'epoch en cours.

def save_last_checkpoint(
    path: Path,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    best_auc: float,
    best_epoch: int,
    epochs_since_best: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "epochs_since_best": epochs_since_best,
        },
        path,
    )


def load_last_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    device: str,
) -> dict[str, Any]:
    """Restaure model/optimizer/scheduler in-place et retourne l'état de reprise.

    Retour : {start_epoch, best_auc, best_epoch, epochs_since_best}
    où start_epoch = epoch de la dernière sauvegarde + 1.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_auc": ckpt["best_auc"],
        "best_epoch": ckpt["best_epoch"],
        "epochs_since_best": ckpt["epochs_since_best"],
    }


# ─── Format durée ────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Seconds → '2 h 47 min 13 s' ou '42 s' / '3 min 15 s'."""
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h} h {m:02d} min {s:02d} s"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"
