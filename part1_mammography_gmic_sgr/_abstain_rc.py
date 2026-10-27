#!/usr/bin/env python
"""Abstention par SEUIL DE CONFIANCE + courbe RISK-COVERAGE (per-breast).

Base = ENSEMBLE-MOYENNE des 5 poids NYU (chacun son percent_t). On infere les
probas malignant par vue sur le set d'ABSTENTION (1/6, jamais vu, sert a CALIBRER)
et sur la VAL (1/6, sert a MESURER), on agrege par sein (moyenne des vues).

Confiance d'un sein : marge au seuil de decision t*  ->  kappa = |p - t*|.
On abstient sur les seins les MOINS confiants (kappa petit = proche de la
frontiere). Protocole honnete : pour chaque couverture cible c, le seuil de marge
tau_c est choisi sur l'ABSTENTION (quantile 1-c de kappa) puis APPLIQUE a la val.

t* = seuil de decision calibre sur l'abstention pour une SENSIBILITE cible
(depistage -> on priorise le rappel des cancers). Defaut 0.90.

Sorties : tableau risk-coverage (val) + CSV + PNG. Probas cachees en .npz."""
import argparse
import sys
import os

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, os.path.join(ROOT, "GMIC"))
sys.path.insert(0, ROOT)
from fine_tuning.dataset_gmic import (load_and_split3, build_samples,
                                      GMICViewDataset, INPUT_SIZE)
from fine_tuning.gmic_finetune import build_gmic, default_parameters
from fine_tuning.train_gmic import gpu_standardize, _side

R = "fine_tuning/checkpoints"
MODELS = [  # (nom, ckpt, percent_t)
    ("model1", R + "/gmic_ft_exam/best.pt", 0.02),
    ("m2", R + "/runs/cancer_malignant_gmic/gmic_nyu-m2_gpu-aug/20260612-132652/best.pt", 0.03),
    ("m3", R + "/runs/cancer_malignant_gmic/gmic_nyu-m3_gpu-aug/20260612-145200/best.pt", 0.03),
    ("m4", R + "/runs/cancer_malignant_gmic/gmic_nyu-m4_gpu-aug/20260612-160747/best.pt", 0.05),
    ("m5", R + "/runs/cancer_malignant_gmic/gmic_nyu-m5_gpu-aug/20260612-172310/best.pt", 0.10),
]
WEIGHTS = "GMIC/models/sample_model_1.p"
IMAGE_DIR = "data/preprocess_image/cropped_images"
PKL = "data/preprocess_image/data.pkl"


def loader_for(samples, batch_size=16, num_workers=6):
    ds = GMICViewDataset(samples, augment=False, input_size=INPUT_SIZE, to_uint8=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, prefetch_factor=2)


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        x = gpu_standardize(x.to(device, non_blocking=True))
        with autocast("cuda"):
            yf, yg, yl, sal = model(x)
        out.extend(yf[:, 1].float().cpu().tolist())
    return np.array(out)


def per_breast(pv, samples):
    from collections import defaultdict
    g, ym = defaultdict(list), {}
    for i, s in enumerate(samples):
        k = (s["pid"], _side(s["view"]))
        g[k].append(pv[i])
        ym[k] = int(s["label"][1])
    keys = list(g.keys())
    return (np.array([float(np.mean(g[k])) for k in keys]),
            np.array([ym[k] for k in keys]))


def ensemble_probs(samples, device):
    """Moyenne (sur les 5 modeles) des probas par vue -> agrege par sein."""
    pv_sum = None
    for name, ck, pt in MODELS:
        params = default_parameters(percent_t=pt, gpu_number=0)
        model = build_gmic(WEIGHTS, params, device)
        sd = torch.load(ck, map_location=device, weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=False)
        loader = loader_for(samples)
        pv = infer(model, loader, device)
        pv_sum = pv if pv_sum is None else pv_sum + pv
        print("  infere " + name + " (pt=" + str(pt) + ")", flush=True)
        del model
        torch.cuda.empty_cache()
    return per_breast(pv_sum / len(MODELS), samples)


def metrics_at(p, y, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    err = (fp + fn) / len(y)
    return dict(sens=sens, spec=spec, err=err, fn=fn, fp=fp,
                npos=int(y.sum()), n=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-target", type=float, default=0.90)
    ap.add_argument("--cache", default="_abstain_probs.npz")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")

    if os.path.exists(args.cache) and not args.recompute:
        d = np.load(args.cache)
        pa, ya, pv, yv = d["pa"], d["ya"], d["pv"], d["yv"]
        print("[cache] " + args.cache)
    else:
        train_ex, abstain_ex, val_ex, _ = load_and_split3(PKL)
        abs_s = build_samples(abstain_ex, IMAGE_DIR)
        val_s = build_samples(val_ex, IMAGE_DIR)
        print("abstention " + str(len(abs_s)) + " vues / val " + str(len(val_s)) + " vues")
        print("== inference ENSEMBLE sur ABSTENTION ==")
        pa, ya = ensemble_probs(abs_s, device)
        print("== inference ENSEMBLE sur VAL ==")
        pv, yv = ensemble_probs(val_s, device)
        np.savez(args.cache, pa=pa, ya=ya, pv=pv, yv=yv)
        print("[cache ecrit] " + args.cache)

    print("\nseins  abstention: %d (%d+)   val: %d (%d+)" %
          (len(ya), int(ya.sum()), len(yv), int(yv.sum())))
    print("AUC per-breast  abstention=%.4f  val=%.4f" %
          (roc_auc_score(ya, pa), roc_auc_score(yv, pv)))

    # t* : seuil de decision pour sensibilite cible, calibre sur l'abstention
    pos = np.sort(pa[ya == 1])
    q = 1.0 - args.sens_target
    tstar = float(np.quantile(pos, q))
    base = metrics_at(pv, yv, tstar)
    print("\nt* (sens cible %.2f sur abstention) = %.4f" % (args.sens_target, tstar))
    print("VAL @100%% couverture, seuil t* : sens=%.3f spec=%.3f err=%.3f (FN=%d, FP=%d, sur %d+ / %d)" %
          (base['sens'], base['spec'], base['err'], base['fn'], base['fp'], base['npos'], base['n']))

    # confiance = marge au seuil ; abstient sur kappa petit
    ka, kv = np.abs(pa - tstar), np.abs(pv - tstar)
    cov_targets = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    print("\n=== RISK-COVERAGE sur VAL (seuil de marge calibre sur l'abstention) ===")
    print("%9s %7s %7s %6s %7s %6s %6s %6s %4s %4s %8s" %
          ("cov_cible", "tau", "cov_val", "n_ret", "pos_ret", "sens", "spec", "err", "FN", "FP", "AUC_ret"))
    rows = []
    for c in cov_targets:
        tau = 0.0 if c >= 0.999 else float(np.quantile(ka, 1.0 - c))
        keep = kv >= tau
        pk, yk = pv[keep], yv[keep]
        covv = keep.mean()
        m = metrics_at(pk, yk, tstar)
        try:
            auc_ret = roc_auc_score(yk, pk) if (yk.sum() > 0 and yk.sum() < len(yk)) else float("nan")
        except ValueError:
            auc_ret = float("nan")
        print("%9.2f %7.4f %7.3f %6d %7d %6.3f %6.3f %6.3f %4d %4d %8.4f" %
              (c, tau, covv, m['n'], m['npos'], m['sens'], m['spec'], m['err'], m['fn'], m['fp'], auc_ret))
        rows.append((c, tau, covv, m['n'], m['npos'], m['sens'], m['spec'], m['err'], m['fn'], m['fp'], auc_ret))

    with open("_abstain_rc.csv", "w") as f:
        f.write("cov_target,tau,cov_val,n_ret,pos_ret,sens,spec,err,fn,fp,auc_ret\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("\nCSV -> _abstain_rc.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        covs = [r[2] for r in rows]
        errs = [r[7] for r in rows]
        aucs = [r[10] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(covs, errs, "o-", color="crimson", label="erreur (retenus)")
        ax1.set_xlabel("couverture (val, fraction decidee)")
        ax1.set_ylabel("erreur", color="crimson")
        ax1.invert_xaxis()
        ax2 = ax1.twinx()
        ax2.plot(covs, aucs, "s--", color="navy", label="AUC retenus")
        ax2.set_ylabel("AUC retenus", color="navy")
        plt.title("Risk-coverage - ensemble 5 NYU, t*@sens%.2f" % args.sens_target)
        fig.tight_layout()
        fig.savefig("_abstain_rc.png", dpi=110)
        print("PNG -> _abstain_rc.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
