#!/usr/bin/env python
"""Rapport de GARANTIES (controle de risque distribution-free, papier Epiconcept).

But: pour chaque metrique, dire ce qu on PEUT GARANTIR (borne certifiee a 1-delta)
et a quelle couverture -- ou NON GARANTISSABLE. On ne regarde pas les courbes
empiriques: uniquement les bornes qui tiennent avec proba 1-delta.

On reutilise le code du papier (selective-classification/python_scripts):
B_star = Prop A1, sgp_dicho = Algo 1, sgp_greedy_search = Algo 2.

On compare DEUX operateurs de decision:
  - seuil 0.5 (argmax)
  - se90    (t* calibre pour ~90% de sensibilite sur les positifs de CALIBRATION)
kappa = max(p, 1-p) (softmax response) dans les deux cas.

Env:
  RC_DELTA  (defaut 0.005 -> 99.5% confiance)
  RC_CACHE  (defaut _abstain_probs.npz) : npz avec pa,ya (calib) et pv,yv (test)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/joshua/selective-classification")
from python_scripts.sgp_utils import sgp_at_targets, emp_metric

DELTA = float(os.environ.get("RC_DELTA", "0.005"))
CACHE = os.environ.get("RC_CACHE", "/home/joshua/breast_cancer/_abstain_probs.npz")
SENS_TARGET = float(os.environ.get("RC_SENS", "0.90"))


def mk(p, y, t):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return pd.DataFrame({
        "p": p,
        "kappa": np.maximum(p, 1 - p),
        "y_pred": (p >= t).astype(int),
        "y_true": np.asarray(y).astype(int),
    })


low_targets = list(np.round(np.arange(0.01, 0.205, 0.005), 4))
high_targets = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

PLAN = [
    ("standard", "dicho",  low_targets,  "<=", "low"),
    ("FP",       "dicho",  low_targets,  "<=", "low"),
    ("FN",       "dicho",  low_targets,  "<=", "low"),
    ("FPR",      "greedy", low_targets,  "<=", "low"),
    ("FNR",      "greedy", low_targets,  "<=", "low"),
    ("SE",       "greedy", high_targets, ">=", "high"),
    ("SP",       "greedy", high_targets, ">=", "high"),
    ("PPV",      "greedy", high_targets, ">=", "high"),
]


def verdict(metric, sense, kind, res):
    if res.empty:
        return "  >>> NON GARANTISSABLE (aucune cible atteinte)"
    if kind == "low":
        best = res.sort_values("metric_bound").iloc[0]
    else:
        best = res.sort_values("metric_bound", ascending=False).iloc[0]
    return ("  >>> GARANTI: %s %s %.4f  @ couverture(test)=%.3f  (theta*=%.4f, "
            "empirique test=%.4f)" % (metric, sense, best.metric_bound,
            best.test_coverage, best.theta_star, best.test_metric))


def run_block(label, train, test, theta_max):
    print("\n" + "#" * 78)
    print("# OPERATEUR: %s   (delta=%.3f -> confiance %.1f%%)" %
          (label, DELTA, 100 * (1 - DELTA)))
    print("#" * 78)
    print("Baselines @100%% couverture:")
    for nm, df in [("calib", train), ("test ", test)]:
        print("  %s n=%d pos=%d pred_pos=%.3f | 0/1=%.4f FP=%.4f FN=%.4f | "
              "FNR=%.3f SE=%.3f FPR=%.3f SP=%.3f PPV=%.3f" %
              (nm, len(df), int(df.y_true.sum()), df.y_pred.mean(),
               emp_metric(df, "standard"), emp_metric(df, "FP"), emp_metric(df, "FN"),
               emp_metric(df, "FNR"), emp_metric(df, "SE"), emp_metric(df, "FPR"),
               emp_metric(df, "SP"), (emp_metric(df, "PPV") if df.y_pred.sum() else np.nan)))

    for metric, mode, targets, sense, kind in PLAN:
        res = sgp_at_targets(train, test, delta=DELTA, metric_targets=targets,
                             metric=metric, mode=mode, theta_min=0.5, theta_max=theta_max)
        print("\n==== %s (%s, garantie %s r*) ====" % (metric, mode, sense))
        if not res.empty:
            cols = ["metric_target", "metric_bound", "theta_star",
                    "train_coverage", "test_metric", "test_coverage"]
            print(res[cols].to_string(index=False, float_format=lambda x: "%.4f" % x))
        print(verdict(metric, sense, kind, res))


def main():
    d = np.load(CACHE)
    pa, ya, pv, yv = d["pa"], d["ya"], d["pv"], d["yv"]
    print("CACHE = %s" % CACHE)
    print("calib %d seins (%d+)  test %d seins (%d+)" %
          (len(ya), int(ya.sum()), len(yv), int(yv.sum())))

    # se90 t* calibre sur les positifs de CALIBRATION (depistage : on vise le rappel)
    pos = np.sort(np.clip(pa[ya == 1], 1e-7, 1 - 1e-7))
    t_se = float(np.quantile(pos, 1.0 - SENS_TARGET))

    for label, t in [("seuil 0.5 (argmax)", 0.5),
                     ("se%d (t*=%.4f)" % (int(SENS_TARGET * 100), t_se), t_se)]:
        train, test = mk(pa, ya, t), mk(pv, yv, t)
        theta_max = float(max(train.kappa.max(), test.kappa.max()))
        run_block(label, train, test, theta_max)


if __name__ == "__main__":
    main()
