"""
Évaluation d'un modèle entraîné depuis son best.pt.

Le chemin doit suivre le format standard du projet :
    fine_tuning/checkpoints/runs/{target}/{model_tag}/{timestamp}/best.pt

La cible (density / normalite / cancer_malignant) et le modèle sont déduits
automatiquement du chemin ou de l'args.json présent dans le même répertoire.

Usage :
    python -m fine_tuning.evaluate <path/to/best.pt>

Exemples :
    python -m fine_tuning.evaluate fine_tuning/checkpoints/runs/density/resnet18_pretrained/20260429-134215/best.pt
    python -m fine_tuning.evaluate fine_tuning/checkpoints/runs/normalite/smallcnn/20260428-162128/best.pt
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─── Constantes ──────────────────────────────────────────────────────────────

DENSITY_CLASSES = ["A", "B", "C", "D"]

BINARY_CLASS_NAMES = {
    "normalite": ["normal", "anormal"],
    "cancer_malignant": ["sain", "cancer"],
}


# ─── Lecture du chemin ───────────────────────────────────────────────────────

def _detect_target_and_model(ckpt_path: Path) -> tuple[str, str]:
    """
    Cherche d'abord dans args.json (source de vérité), puis parse le chemin.
    Format attendu : .../runs/{target}/{model_tag}/{timestamp}/best.pt
    """
    args_path = ckpt_path.parent / "args.json"
    if args_path.exists():
        args = json.loads(args_path.read_text())
        return args.get("target", "?"), args.get("model_arch", "?")

    parts = ckpt_path.parts
    try:
        runs_idx = next(i for i, p in enumerate(parts) if p == "runs")
        target    = parts[runs_idx + 1]
        model_tag = parts[runs_idx + 2]
        return target, model_tag
    except (StopIteration, IndexError):
        raise ValueError(
            f"Impossible de détecter la cible depuis : {ckpt_path}\n"
            "Assure-toi que le chemin contient '.../runs/{target}/{model_tag}/...'"
        )


# ─── Évaluation density (multi-classe) ───────────────────────────────────────

def _eval_density(ckpt: dict, run_dir: Path) -> None:
    preds   = ckpt["val_preds"]    # list[int] — argmax déjà calculé pendant le train
    targets = ckpt["val_targets"]  # list[int]
    epoch   = ckpt["epoch"]
    best_f1 = ckpt.get("val_macro_f1", None)

    accuracy = ckpt.get("val_accuracy", float("nan"))
    report = classification_report(
        targets, preds,
        labels=list(range(len(DENSITY_CLASSES))),
        target_names=DENSITY_CLASSES,
        digits=3,
    )

    header = (
        f"Run     : {run_dir}\n"
        f"Epoch best  : {epoch}\n"
        f"Macro-F1    : {best_f1:.4f}\n"
        f"Accuracy    : {accuracy:.4f}\n"
        f"\n{report}"
    )
    print(header)

    out_txt = run_dir / "eval_report.txt"
    out_txt.write_text(header)
    print(f"Rapport sauvé → {out_txt}")

    cm = confusion_matrix(targets, preds, labels=list(range(len(DENSITY_CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=DENSITY_CLASSES, yticklabels=DENSITY_CLASSES, ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité")
    ax.set_title(f"Matrice de confusion — density (epoch {epoch}, macro-F1={best_f1:.3f})")
    plt.tight_layout()
    out = run_dir / "eval_confusion_matrix.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Matrice de confusion sauvée → {out}")
    plt.show()


# ─── Évaluation binaire (normalite / cancer_malignant) ───────────────────────

def _eval_binary(ckpt: dict, target: str, run_dir: Path) -> None:
    probs   = np.array(ckpt["val_preds"])    # float — probabilités sigmoid
    targets = np.array(ckpt["val_targets"])  # 0/1
    epoch   = ckpt["epoch"]
    best_auc = ckpt.get("val_auc", None)

    preds_bin = (probs >= 0.5).astype(int)
    class_names = BINARY_CLASS_NAMES.get(target, ["négatif", "positif"])

    auc = roc_auc_score(targets, probs)
    report = classification_report(targets, preds_bin, target_names=class_names, digits=3)

    header = (
        f"Run     : {run_dir}\n"
        f"Epoch best  : {epoch}\n"
        f"AUC val     : {auc:.4f}" + (f"  (enregistré : {best_auc:.4f})" if best_auc else "") + "\n"
        f"\n{report}"
    )
    print(header)

    out_txt = run_dir / "eval_report.txt"
    out_txt.write_text(header)
    print(f"Rapport sauvé → {out_txt}")

    fpr, tpr, _ = roc_curve(targets, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"Courbe ROC — {target} (epoch {epoch})")
    ax.legend()
    plt.tight_layout()
    out = run_dir / "eval_roc.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Courbe ROC sauvée → {out}")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    if not ckpt_path.exists():
        print(f"Fichier introuvable : {ckpt_path}")
        sys.exit(1)

    target, model_tag = _detect_target_and_model(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_dir = ckpt_path.parent

    print("=" * 60)
    print(f"Run     : {run_dir}")
    print(f"Cible   : {target}")
    print(f"Modèle  : {model_tag}")
    print("=" * 60)
    print()

    if target == "density":
        _eval_density(ckpt, run_dir)
    elif target in ("normalite", "cancer_malignant"):
        _eval_binary(ckpt, target, run_dir)
    else:
        print(f"Cible inconnue : '{target}'. Attendu : density / normalite / cancer_malignant")
        sys.exit(1)


if __name__ == "__main__":
    main()
