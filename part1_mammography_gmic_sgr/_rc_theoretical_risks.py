#!/usr/bin/env python
"""Risques THEORIQUES conditionnels : FNR & FPR uniquement.

On ne traite QUE les deux risques conditionnels (non directement observables) :
  - FPR = P(predire cancer | vrai = sain)      -> conditionne sur Y=0
  - FNR = P(predire sain   | vrai = cancer)     -> conditionne sur Y=1
On ne touche PAS aux risques 'pratiques'/marginaux (0/1, comptes FP/FN, SE...).

But : montrer la borne GARANTIE (distribution-free) de chaque risque et POURQUOI
on peut la garantir (FPR) ou non (FNR). Meme formule pour les deux :

    borne = b / (proportion_classe_conditionnement - d2)

ou b = borne haute sur les erreurs (B_star, Prop A1), et d2 = marge de
concentration. La seule difference FPR vs FNR : la classe de conditionnement
est abondante (sains ~98%) ou rare (cancers ~2%).

Code du papier : ~/selective-classification/python_scripts/sgp_utils.py
"""
import os, sys, math
import numpy as np, pandas as pd

sys.path.insert(0, "/home/joshua/selective-classification")
from python_scripts.sgp_utils import (sgp_at_targets, emp_metric,
                                       upper_bound_denominator, our_bound)

CACHE = os.environ.get("RC_CACHE", "/home/joshua/breast_cancer/_abstain_probs_bbdrop.npz")
DELTA = float(os.environ.get("RC_DELTA", "0.005"))


def mk(p, y, t=0.5):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return pd.DataFrame({"p": p, "kappa": np.maximum(p, 1 - p),
                         "y_pred": (p >= t).astype(int), "y_true": np.asarray(y).astype(int)})


def main():
    d = np.load(CACHE)
    cal, test = mk(d["pa"], d["ya"]), mk(d["pv"], d["yv"])
    n = len(cal)
    print("CACHE=%s" % CACHE)
    print("calib n=%d  cancers(Y=1)=%d  sains(Y=0)=%d  | delta=%g (confiance %.1f%%)\n"
          % (n, int(cal.y_true.sum()), int((1 - cal.y_true).sum()), DELTA, 100 * (1 - DELTA)))

    print("=== Decomposition pleine couverture : borne = b / (prop_classe_cond - d2) ===\n")
    d2 = math.sqrt(n * math.log(2 / DELTA) / 2) / n
    for metric, cond, classe in [("FPR", "Y=0", "sains"), ("FNR", "Y=1", "cancers")]:
        emp = emp_metric(cal, metric)
        denom = upper_bound_denominator(metric, cal, DELTA, n)   # = prop_classe_cond - d2
        prop = denom + d2                                         # proportion de la classe de cond.
        B = our_bound(cal, metric, n, delta=DELTA)
        verdict = ("GARANTI : %s <= %.3f" % (metric, B)) if (B == B) else \
                  ("NON GARANTISSABLE (denom=%.3f trop petit -> borne hors [0,1])" % denom)
        print("[%s] conditionne sur %s (%s)" % (metric, cond, classe))
        print("   empirique=%.3f | proportion %s=%.3f  d2=%.3f  denom=%.3f" %
              (emp, classe, prop, d2, denom))
        print("   -> %s\n" % verdict)

    print("=== Meilleure borne atteignable avec abstention (sgp, Algo 2) ===")
    tmax = float(max(cal.kappa.max(), test.kappa.max()))
    for metric in ["FPR", "FNR"]:
        r = sgp_at_targets(cal, test, delta=DELTA, metric=metric, mode="greedy",
                           metric_targets=[i / 100 for i in range(1, 51)],
                           theta_min=0.5, theta_max=tmax)
        if r.empty:
            print("  %s : NON GARANTISSABLE" % metric)
        else:
            b = r.sort_values("metric_bound").iloc[0]
            print("  %s : GARANTI <= %.3f @ couverture(test)=%.3f" %
                  (metric, b.metric_bound, b.test_coverage))


if __name__ == "__main__":
    main()
