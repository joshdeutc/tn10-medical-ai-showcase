"""
Adaptateur GMIC → sgp_set : branche une inférence GMIC sur la chaîne d'abstention.

L'inférence GMIC (scripts/inference.py) produit un `predictions.csv` :
    image_index, malignant_pred, benign_pred, malignant_label

Ce script le convertit au format `sgp_set` attendu par abstention_module
(mêmes colonnes que `infer.py` : image_id, split, y_true, y_pred, p, kappa),
écrit le run dans abstention_module/runs/ avec un meta.json identique aux autres
(diagnostic + baselines via les helpers d'infer.py), puis tu peux tracer la
courbe accuracy vs seuil avec le code existant, SANS rien réécrire :

    uv run python scripts/eval_gmic.py
    uv run python -m abstention_module.threshold --run cancer/gmic-nyu-sample1/<ts> --split cal

Convention de conversion (cf. infer.py:_run_inference) :
    p      = malignant_pred                 (proba de la classe positive = malignité)
    y_pred = (p >= 0.5)
    kappa  = max(p, 1 - p)                  (softmax response = confiance)
    y_true = malignant_label                (vérité terrain : 1 = cancer)

Le split cal/test (75/25, seed 42) reproduit le découpage des autres runs RSNA.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_module.infer import (
    PROJECT_ROOT,
    RUNS_DIR,
    _abstention_diagnostic,
    _summary_stats,
)

DEFAULT_PRED = PROJECT_ROOT / "data/preprocess_image/rsna_output/predictions.csv"
GMIC_CKPT = "GMIC/models/sample_model_1.p"
TEST_FRACTION = 0.25
SEED = 42


def build_sgp(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    """predictions.csv GMIC → sgp_set (image_id, split, y_true, y_pred, p, kappa)."""
    p = df[score_col].astype(float).to_numpy()
    y_true = df[label_col].astype(int).to_numpy()
    y_pred = (p >= 0.5).astype(int)
    kappa = np.maximum(p, 1.0 - p)

    rng = np.random.default_rng(SEED)
    is_test = rng.random(len(df)) < TEST_FRACTION
    split = np.where(is_test, "test", "cal")

    return pd.DataFrame(
        {
            "image_id": df["image_index"].astype(str).to_numpy(),
            "split": split,
            "y_true": y_true,
            "y_pred": y_pred,
            "p": p,
            "kappa": kappa,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--score-col", default="malignant_pred")
    ap.add_argument("--label-col", default="malignant_label")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    sgp = build_sgp(df, args.score_col, args.label_col)

    inference_ts = datetime.fromtimestamp(args.predictions.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    # Arborescence imbriquée runs/<target>/<model_tag>/<timestamp>/ (cf. infer.py / fine_tuning)
    name = args.run_name or f"cancer/gmic-nyu-sample1/{inference_ts}"
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sgp.to_pickle(out_dir / "sgp_set.pkl")
    sgp.to_csv(out_dir / "sgp_set.csv", index=False)

    diag = _abstention_diagnostic(sgp)
    meta = {
        "name": name,
        "run_kind": "gmic_evaluation",
        "description": (
            "sgp_set construit à partir des prédictions GMIC (modèle pré-entraîné NYU, "
            "AUCUN fine-tuning ici) — pas de checkpoint maison. La colonne `p` = malignant_pred."
        ),
        "target": "cancer",
        "dataset_name": "RSNA Breast Cancer Detection (2022, Kaggle)",
        "dataset_kind": "rsna",
        "model_tag": "gmic-nyu-sample1",
        "training_timestamp": "nyu-pretrained",
        "abstention_diagnostic": diag,
        "kappa_min": round(float(sgp["kappa"].min()), 6),
        "kappa_max": round(float(sgp["kappa"].max()), 6),
        "columns": {
            "p": "malignant_pred — proba GMIC de malignité (classe positive)",
            "kappa": "softmax response = max(p, 1-p) — confiance, sert de seuil d'abstention",
            "y_pred": "1 si p >= 0.5",
            "y_true": "malignant_label — vérité terrain (1 = cancer)",
            "split": f"cal/test aléatoire {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)} (seed {SEED})",
        },
        "inference": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_predictions": str(args.predictions.relative_to(PROJECT_ROOT)),
            "predictions_mtime": datetime.fromtimestamp(
                args.predictions.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "script": "scripts/inference.py",
            "model_index": "1",
            "image_dir": "data/preprocess_image/rsna_output",
            "dataset_kind": "rsna",
        },
        "checkpoint": {
            "path": GMIC_CKPT,
            "arch": "ScratchGMIC (K=6, crop 256x256, percent_t=0.02, num_classes=2)",
            "best_auc": None,
            "note": "poids NYU pré-entraînés, non re-entraînés sur RSNA",
        },
        "dataset": {
            "n_total": int(len(sgp)),
            "split_sizes": {k: int(v) for k, v in sgp["split"].value_counts().to_dict().items()},
        },
        "baselines": _summary_stats(sgp),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"✓ sgp_set écrit → abstention_module/runs/{name}/")
    print(f"  n={len(sgp)}  | kappa∈[{sgp.kappa.min():.3f}, {sgp.kappa.max():.3f}]")
    for split, m in meta["baselines"].items():
        print(f"  {split:>4}: n={m['n']:>5}  pos={m['positives']:>4} ({m['base_rate']*100:.1f}%)  "
              f"acc@0.5={m['accuracy_at_0.5']*100:5.2f}%  pred_pos={m['pred_positive_rate']*100:5.2f}%")
    print(f"  abstention utilisable : {diag['usable_for_abstention']}  flags={diag['pathology_flags']}")
    print(f"\nCourbe accuracy vs seuil :")
    print(f"  uv run python -m abstention_module.threshold --run {name} --split cal")


if __name__ == "__main__":
    main()
