#!/usr/bin/env python
"""Construit le sgp_set.pkl + meta.json de l'ENSEMBLE 5 NYU pour algo1/algo2.

Format attendu par le module (cf meta du run de reference) :
  p       : proba malignant d'ensemble (moyenne des 5)
  kappa   : softmax response = max(p, 1-p)  (confiance, seuil d'abstention)
  y_pred  : 1(p >= 0.5)
  y_true  : label malignant
  split   : cal/test (informatif ; algo1/algo2 re-splittent eux-memes 75/25)

Donnees = abstention (1/6) UNION val (1/6) : tout est held-out de l'entrainement
GMIC. algo1/algo2 decoupent ensuite bound-fitting/test en interne par seed.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/joshua/breast_cancer")

# Mode de seuil de decision f(x)=1(p>=t*) + definition de la confiance kappa.
#   "argmax"  : t*=0.5, kappa = SR = max(p,1-p)            (setup fidele du module)
#   "se<NN>"  : t* calibre pour sensibilite NN% sur les positifs de calibration,
#               kappa = 0.5 + 0.5*|p-t*|/max(t*,1-t*) (recentree sur t*, dans [0.5,1]
#               -> compatible avec la grille theta [0.5, kappa_max] de algo1/algo2)
MODE = os.environ.get("RC_THRESH", "argmax")

d = np.load(ROOT / "_abstain_probs.npz")
pa, ya, pv, yv = d["pa"], d["ya"], d["pv"], d["yv"]

if MODE == "argmax":
    tstar = 0.5
    tag = "gmic-ens5-nyu"

    def kappa_of(p):
        return np.maximum(p, 1 - p)
elif MODE.startswith("se"):
    sens = int(MODE[2:]) / 100.0
    pos = np.sort(np.clip(pa[ya == 1], 1e-7, 1 - 1e-7))
    tstar = float(np.quantile(pos, 1.0 - sens))   # calibre sur l'abstention
    tag = "gmic-ens5-nyu-%s" % MODE

    def kappa_of(p):  # SR standard du module (indep. du seuil) ; seul y_pred change
        return np.maximum(p, 1 - p)
else:
    raise SystemExit("RC_THRESH inconnu: " + MODE)

RUN_ID = "cancer/" + tag
ts = time.strftime("%Y%m%d-%H%M%S")
run_dir = ROOT / "abstention_module" / "runs" / RUN_ID / ts
run_dir.mkdir(parents=True, exist_ok=True)

frames = []
for split, p, y in [("cal", pa, ya), ("test", pv, yv)]:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    frames.append(pd.DataFrame({
        "p": p,
        "kappa": kappa_of(p),
        "y_pred": (p >= tstar).astype(int),
        "y_true": np.asarray(y).astype(int),
        "split": split,
    }))
sgp = pd.concat(frames, ignore_index=True)
print("MODE =", MODE, "| t* =", round(tstar, 4))

sgp.to_pickle(run_dir / "sgp_set.pkl")
sgp.to_csv(run_dir / "sgp_set.csv", index=False)

meta = {
    "name": f"{RUN_ID}/{ts}",
    "run_kind": "gmic_ensemble_evaluation",
    "description": "Ensemble-moyenne des 5 GMIC finetunes NYU (sample_model_1..5, "
                   "percent_t propres). sgp_set = abstention(1/6) UNION val(1/6), "
                   "per-breast. p=proba malignant d'ensemble.",
    "target": "cancer",
    "dataset_kind": "rsna",
    "model_tag": tag,
    "decision": {"mode": MODE, "t_star": float(tstar)},
    "columns": {
        "p": "malignant_pred d'ensemble (moyenne des 5)",
        "kappa": "softmax response = max(p, 1-p)",
        "y_pred": "1 si p >= 0.5",
        "y_true": "malignant_label",
        "split": "cal(abstention) / test(val) — informatif",
    },
    "kappa_min": float(sgp.kappa.min()),
    "kappa_max": float(sgp.kappa.max()),
    "dataset": {
        "n_total": int(len(sgp)),
        "split_sizes": {k: int(v) for k, v in sgp.split.value_counts().items()},
        "positives_total": int(sgp.y_true.sum()),
        "base_rate": float(sgp.y_true.mean()),
        "pred_positive_rate_at_0.5": float(sgp.y_pred.mean()),
    },
}
(run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# lien latest
latest = ROOT / "abstention_module" / "runs" / "latest"
try:
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir)
except OSError:
    pass

print("RUN_ID =", f"{RUN_ID}/{ts}")
print("run_dir =", run_dir)
print("n_total =", len(sgp), "| positives =", int(sgp.y_true.sum()),
      "| kappa in [%.4f, %.4f]" % (sgp.kappa.min(), sgp.kappa.max()))
