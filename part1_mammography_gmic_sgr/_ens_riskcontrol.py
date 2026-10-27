#!/usr/bin/env python
"""Controle de risque distribution-free (papier Epiconcept) sur l'ENSEMBLE 5 NYU.

Utilise le module abstention_module DEJA implemente (B_star = Prop A1, sgp_dicho
= Algo 1, sgp_greedy_search = Algo 2, sgp_at_targets). On ne reimplemente rien :
on alimente ces fonctions avec les probas d'ensemble deja en cache.

  train = set d'ABSTENTION (1/6, jamais vu a l'entrainement NI a la selection)
          -> sert a CALIBRER le seuil + fournit la garantie.
  test  = VAL (1/6) -> verifie empiriquement que la garantie tient.

f(x) = 1(p >= 0.5) (argmax), kappa = max(p, 1-p) (softmax response), comme dans
le run de reference du module. delta = 0.05 -> garantie a 95 % de confiance.
"""
import sys
import numpy as np
import pandas as pd

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, ROOT)
from abstention_module.sgp_utils import sgp_at_targets, emp_metric  # noqa

import os
DELTA = float(os.environ.get("RC_DELTA", "0.005"))  # defaut 99.5 % (papier)


def mk(p, y):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return pd.DataFrame({
        "p": p,
        "kappa": np.maximum(p, 1 - p),
        "y_pred": (p >= 0.5).astype(int),
        "y_true": np.asarray(y).astype(int),
    })


def main():
    d = np.load("_abstain_probs.npz")
    train = mk(d["pa"], d["ya"])   # abstention = calibration
    test = mk(d["pv"], d["yv"])    # val = test
    theta_max = float(max(train.kappa.max(), test.kappa.max()))

    print("=== Baselines @100%% couverture (seuil 0.5) ===")
    for nm, df in [("calib(abstention)", train), ("test(val)", test)]:
        print("  %-18s n=%d pos=%d  pred_pos_rate=%.4f | 0/1=%.4f FP=%.4f FN=%.4f | "
              "FNR=%.3f SE=%.3f FPR=%.3f SP=%.3f" %
              (nm, len(df), int(df.y_true.sum()), df.y_pred.mean(),
               emp_metric(df, "standard"), emp_metric(df, "FP"), emp_metric(df, "FN"),
               emp_metric(df, "FNR"), emp_metric(df, "SE"),
               emp_metric(df, "FPR"), emp_metric(df, "SP")))

    # cibles : metriques "a borner par le haut" (low) vs "a garantir par le bas" (high)
    low_targets = np.round(np.arange(0.01, 0.205, 0.005), 4)   # inclut 0.02
    high_targets = np.round(np.array([0.80, 0.85, 0.90, 0.95, 0.98, 0.99]), 4)

    plan = [
        ("standard", "dicho", low_targets, "<= r*"),
        ("FP", "dicho", low_targets, "<= r*"),
        ("FN", "dicho", low_targets, "<= r*"),
        ("FPR", "greedy", low_targets, "<= r*"),
        ("FNR", "greedy", low_targets, "<= r*"),
        ("SE", "greedy", high_targets, ">= r*"),
        ("SP", "greedy", high_targets, ">= r*"),
        ("PPV", "greedy", high_targets, ">= r*"),
    ]

    for metric, mode, targets, sense in plan:
        res = sgp_at_targets(train, test, delta=DELTA, metric_targets=list(targets),
                             metric=metric, mode=mode, theta_min=0.5, theta_max=theta_max)
        print("\n==== %s  (%s, garantie %s, delta=%.3f) ====" % (metric, mode, sense, DELTA))
        if res.empty:
            print("   aucune cible atteignable sur ce set")
            continue
        cols = ["metric_target", "metric_bound", "theta_star",
                "train_metric", "train_coverage", "test_metric", "test_coverage"]
        print(res[cols].to_string(index=False,
              float_format=lambda x: "%.4f" % x))

    print("\n--- REPONSE DIRECTE : 'se tromper <= 2/100 a 95%% de confiance' ---")
    res = sgp_at_targets(train, test, delta=DELTA, metric_targets=[0.02],
                         metric="standard", mode="dicho", theta_min=0.5, theta_max=theta_max)
    if res.empty:
        print("  Pas atteignable: meme en abstenant beaucoup, on ne CERTIFIE pas 2%% (0/1).")
    else:
        r = res.iloc[0]
        print("  theta* = %.4f  -> on garde les seins de confiance kappa >= theta*" % r.theta_star)
        print("  borne 0/1 GARANTIE (95%%) = %.4f  (<= 0.02 demande)" % r.metric_bound)
        print("  couverture calibration = %.3f  | couverture test(val) = %.3f" %
              (r.train_coverage, r.test_coverage))
        print("  risque 0/1 empirique sur val (retenus) = %.4f" % r.test_metric)


if __name__ == "__main__":
    main()
