#!/usr/bin/env python
"""
Finetuning complet de GMIC sur RSNA (cancer du sein), full-res 2944x1920.

Loss (article GMIC, reconstruite a partir des 3 tetes) :
    L = BCE(y_fusion) + BCE(y_global) + BCE(y_local) + beta * ||saliency||_1

ATTENTION : les 3 sorties sont DEJA passees en sigmoide -> on utilise
binary_cross_entropy (sur probas), JAMAIS BCEWithLogits. La BCE est calculee
en float32 HORS du contexte autocast (binary_cross_entropy n'est pas autocast-safe).

Strategie : full d'emblee, LR discriminatif (backbone bas, tetes haut).
Sortie : 2 canaux [benign, malignant] supervises (lancer fix_benign_labels.py avant).

Selection & arret : on entraine par VUE (GMIC traite une vue a la fois) mais on
SELECTIONNE et on fait l'EARLY STOPPING sur l'AUC_malignant agregee au niveau
du SEIN (pid + cote, moyenne des vues CC+MLO) -> unite cliniquement etiquetee
(RSNA), pas la vue isolee. best.pt = meilleur AUC_mal niveau examen.

Augmentation : --gpu-augment deplace l'augmentation (affine + gamma) sur GPU
via kornia -> les workers CPU ne font plus que decoder les PNG (la machine est
CPU/IO-bound a 8 vCPU, pas GPU-bound). Sans le flag : augmentation CPU du
dataset (comportement historique).

Organisation : chaque run cree runs/{target}/{model_tag}/{timestamp}/ avec
args.json (verite machine) + README.md humain (mis a jour en fin de run),
via run_metadata. --out court-circuite si fourni (retro-compat).

Usage :
    ./.venv/bin/python -m fine_tuning.train_gmic \
        --weights GMIC/models/sample_model_1.p --percent-t 0.02 \
        --gpu-augment --abstain-frac 0 --model-tag gmic_full-ft_gpu-aug \
        2>&1 | tee /tmp/train_gmic.log   # le vrai log vit dans le run dir
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GMIC/ doit etre dans sys.path pour que `from src.modeling...` resolve
sys.path.insert(0, os.path.expanduser("~/GMIC"))  # GMIC vit dans le HOME (pas dans le repo)
sys.path.insert(0, PROJECT_ROOT)

from fine_tuning.dataset_gmic import make_loaders, INPUT_SIZE, _side  # noqa: E402
from fine_tuning.gmic_finetune import (  # noqa: E402
    build_gmic, default_parameters, param_groups, freeze_backbone, set_backbone_eval,
    freeze_global, set_global_eval, add_backbone_dropout,
)
from fine_tuning.run_metadata import (  # noqa: E402
    make_run_dir, get_git_commit, write_run_readme, write_args_json, format_duration,
)
# GPUAugment + gpu_standardize vivent desormais dans fine_tuning/augment.py
# (re-exportes ici pour ne casser aucun import existant).
from fine_tuning.augment import GPUAugment, gpu_standardize  # noqa: E402,F401

EPS = 1e-7


def bce(prob, target):
    prob = prob.clamp(EPS, 1.0 - EPS)
    return nn.functional.binary_cross_entropy(prob, target)


def compute_loss(y_fusion, y_global, y_local, saliency, target, beta):
    # forcer fp32 (autocast deja quitte au moment de l'appel)
    l_fus = bce(y_fusion.float(), target)
    l_glo = bce(y_global.float(), target)
    l_loc = bce(y_local.float(), target)
    l_reg = beta * saliency.float().abs().mean() if beta > 0 else 0.0
    total = l_fus + l_glo + l_loc + l_reg
    return total, {
        "loss": float(total.detach()),
        "bce_fusion": float(l_fus.detach()),
        "bce_global": float(l_glo.detach()),
        "bce_local": float(l_loc.detach()),
        "l1_saliency": float(l_reg.detach()) if beta > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, samples, device, beta):
    """Evalue par VUE puis agrege par SEIN (exam-level RSNA : pid + cote).

    `samples` doit etre la liste des echantillons du loader, dans le MEME ordre
    (loader val : shuffle=False, pas de drop_last -> ordre = ordre du dataset).
    L'AUC "exam" est calculee au niveau du sein : pour chaque (pid, cote) on
    moyenne les probas des vues (CC + MLO), unite cliniquement etiquetee.
    """
    model.eval()
    preds_b, preds_m, preds_gm, preds_lm, labs = [], [], [], [], []
    running = 0.0
    n = 0
    for x, y in loader:
        x = gpu_standardize(x.to(device, non_blocking=True))
        yt = y.to(device)
        with autocast("cuda"):
            yf, yg, yl, sal = model(x)
        total, _ = compute_loss(yf, yg, yl, sal, yt, beta)
        running += float(total) * x.size(0)
        n += x.size(0)
        yf = yf.float().cpu().numpy()
        yg = yg.float().cpu().numpy()
        yl = yl.float().cpu().numpy()
        preds_b.extend(yf[:, 0].tolist())
        preds_m.extend(yf[:, 1].tolist())
        preds_gm.extend(yg[:, 1].tolist())  # malignant, tete GLOBAL
        preds_lm.extend(yl[:, 1].tolist())  # malignant, tete LOCAL
        labs.append(y.numpy())
    labs = np.concatenate(labs)
    preds_m = np.array(preds_m)
    preds_b = np.array(preds_b)
    preds_gm = np.array(preds_gm)
    preds_lm = np.array(preds_lm)

    def safe_auc(y_true, y_score):
        y_true = np.asarray(y_true)
        return roc_auc_score(y_true, y_score) if (y_true.sum() > 0 and y_true.sum() < len(y_true)) else float("nan")

    # --- agregation niveau SEIN (pid, cote) : moyenne des probas des vues ---
    assert len(samples) == len(preds_m), (
        f"desalignement samples({len(samples)}) vs preds({len(preds_m)}) : "
        "le loader val doit etre sequentiel (shuffle=False, drop_last=False)"
    )
    from collections import defaultdict
    g_m, g_b, g_gm, g_lm, g_ym, g_yb = (defaultdict(list), defaultdict(list),
        defaultdict(list), defaultdict(list), {}, {})
    for i, s in enumerate(samples):
        key = (s["pid"], _side(s["view"]))
        g_m[key].append(preds_m[i])
        g_b[key].append(preds_b[i])
        g_gm[key].append(preds_gm[i])
        g_lm[key].append(preds_lm[i])
        g_ym[key] = int(s["label"][1])
        g_yb[key] = int(s["label"][0])
    keys = list(g_m.keys())
    exam_pm = np.array([float(np.mean(g_m[k])) for k in keys])
    exam_pb = np.array([float(np.mean(g_b[k])) for k in keys])
    exam_pgm = np.array([float(np.mean(g_gm[k])) for k in keys])
    exam_plm = np.array([float(np.mean(g_lm[k])) for k in keys])
    exam_ym = np.array([g_ym[k] for k in keys])
    exam_yb = np.array([g_yb[k] for k in keys])

    return {
        "val_loss": running / max(n, 1),
        "auc_malignant": safe_auc(labs[:, 1], preds_m),
        "auc_benign": safe_auc(labs[:, 0], preds_b),
        "exam_auc_malignant": safe_auc(exam_ym, exam_pm),
        "exam_auc_benign": safe_auc(exam_yb, exam_pb),
        "auc_global_mal": safe_auc(labs[:, 1], preds_gm),
        "auc_local_mal": safe_auc(labs[:, 1], preds_lm),
        "exam_auc_global_mal": safe_auc(exam_ym, exam_pgm),
        "exam_auc_local_mal": safe_auc(exam_ym, exam_plm),
        "n_exams": len(keys),
    }


def save_ckpt(path, epoch, model, optimizer, scheduler, scaler, best_auc, logs):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_auc": best_auc,
        "logs": logs,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/preprocess_image/data.pkl")
    ap.add_argument("--image-dir", default="data/preprocess_image/cropped_images")
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p")
    ap.add_argument("--percent-t", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=40, help="plafond d'epochs (early stopping coupe avant en general)")
    ap.add_argument("--patience", type=int, default=6,
                    help="early stopping : nb d'epochs sans amelioration de l'AUC_mal niveau examen avant arret")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="amelioration minimale d'AUC pour reset le compteur d'early stopping")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=1e-4, help="poids L1 sur la saliency")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout sur l'entree du classifieur fusion (train-only)")
    ap.add_argument("--backbone-dropout", type=float, default=0.0, help="dropout2d (spatial, train-only) DANS les backbones global ds_net + local dn_resnet")
    ap.add_argument("--freeze-backbone", action="store_true", help="linear-probe : gele ds_net+dn_resnet, entraine seulement les tetes")
    ap.add_argument("--freeze-global", action="store_true", help="gele SEULEMENT le backbone global (ds_net) ; dn_resnet + tetes entraines (diagnostic par tete)")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=2.0 / 3,
                    help="part patients pour l'entrainement GMIC")
    ap.add_argument("--abstain-frac", type=float, default=1.0 / 6,
                    help="part patients reservee au modele d'abstention (held-out)")
    ap.add_argument("--val-frac", type=float, default=1.0 / 6,
                    help="part patients pour la validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gpu-augment", action="store_true",
                    help="augmentation kornia sur GPU (les workers CPU ne font plus que decoder)")
    ap.add_argument("--runs-root", default="fine_tuning/checkpoints/runs",
                    help="racine de l'arborescence runs/{target}/{model_tag}/{timestamp}")
    ap.add_argument("--model-tag", default=None,
                    help="libelle du run (defaut auto : gmic_{full-ft|frozen}_{gpu|cpu}-aug)")
    ap.add_argument("--out", default=None,
                    help="court-circuite l'arborescence runs/ (retro-compat anciens scripts)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    use_amp = not args.no_amp

    # --- run dir documente : runs/{target}/{model_tag}/{timestamp}/ ---
    target = "cancer_malignant_gmic"
    if args.model_tag is None:
        _mode = ("frozen" if args.freeze_backbone else
                 "freeze-global" if args.freeze_global else "full-ft")
        args.model_tag = ("gmic_" + _mode
                          + ("_gpu-aug" if args.gpu_augment else "_cpu-aug")
                          + (f"_bbdrop{args.backbone_dropout}" if args.backbone_dropout > 0 else ""))
    if args.out:
        run_dir = Path(args.out)            # retro-compat
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(Path(args.runs_root), target, args.model_tag)
    args.out = str(run_dir)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t_run0 = time.time()

    # stdout/stderr dupliques dans run_dir/train.log (plus besoin de tee externe,
    # le timestamp du run dir n'est pas connaissable avant le lancement)
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()

    _logf = open(run_dir / "train.log", "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, _logf)
    sys.stderr = _Tee(sys.stderr, _logf)

    print("=" * 60)
    print(f"Finetuning GMIC | run dir : {run_dir}")
    for k, v in vars(args).items():
        print(f"  {k:14s}: {v}")
    print("=" * 60)

    # --- data ---
    # GPU-augment actif -> augmentation CPU du dataset desactivee (les workers
    # ne font plus que decoder les PNG)
    train_loader, val_loader, train_samples, val_samples = make_loaders(
        pkl_path=args.pkl, image_dir=args.image_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_frac=args.train_frac, abstain_frac=args.abstain_frac,
        val_frac=args.val_frac, seed=args.seed, augment=not args.gpu_augment,
        input_size=INPUT_SIZE,
        split_out=os.path.join(args.out, "split_patients.json"),
    )
    gpu_aug = GPUAugment(input_size=INPUT_SIZE).to(device) if args.gpu_augment else None

    # profondeur du cache image (uint8 8 bits vs uint16 natif 12 bits)
    import imageio.v2 as _imageio
    pixel_depth = str(np.asarray(_imageio.imread(train_samples[0]["path"])).dtype)
    print(f"[data] profondeur du cache {args.image_dir} : {pixel_depth}")

    # --- metadonnees du run (README humain + args.json machine) ---
    aug_desc = ("kornia GPU : affine (rot +/-15deg, scale 0.9-1.1, transl +/-48/32px, "
                "fond noir) + gamma 0.8-1.25 p=0.5 ; rotation ROI {0,90,180,270} dans "
                "le forward (train only) ; PAS de hflip" if args.gpu_augment else
                "CPU (dataset._augment) : affine + gamma ; rotation ROI dans le forward")
    meta = {
        "target": target,
        "aggregation": "entrainement par vue ; selection/early-stop par SEIN (pid+cote, mean CC+MLO)",
        "model_arch": f"GMIC (ResNet-22 global + ResNet-18 local, K=6 patchs 256x256, percent_t={args.percent_t})",
        "pretrained": True,
        "init_desc": f"poids NYU pre-entraines `{args.weights}` (strict=False)",
        "head_desc": "3 tetes sigmoidees [benign, malignant] : y_global (top-t%), y_local (MIL attention), y_fusion",
        "dataset_name": f"RSNA breast cancer (data.pkl, labels benign corriges via biopsy, cache {pixel_depth})",
        "image_dir": args.image_dir,
        "img_size": f"2944×1920 (H×W, full-res GMIC, {pixel_depth})",
        "pixel_depth": pixel_depth,
        "split_desc": (f"patient-level stratifie seed={args.seed} : "
                       f"train {args.train_frac:.0%} / abstention {args.abstain_frac:.0%} / "
                       f"val {args.val_frac:.0%}" +
                       (" (PAS de held-out abstention : tout sauf val en train)"
                        if args.abstain_frac <= 0 else "")),
        "random_seed": args.seed,
        "n_train": len(train_samples),
        "n_positive_train": int(sum(s["malignant"] for s in train_samples)),
        "n_val": len(val_samples),
        "n_positive_val": int(sum(s["malignant"] for s in val_samples)),
        "started_at": started_at,
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": f"backbone {args.lr_backbone} / tetes {args.lr_head}",
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "augmentation": aug_desc,
        "sampler": "WeightedRandomSampler ~50/50 vues avec lesion vs negatives (recette papier)",
        "git_commit": get_git_commit(Path(PROJECT_ROOT)),
        "loss": f"BCE(y_fusion)+BCE(y_global)+BCE(y_local)+{args.beta}*L1(saliency)",
        "files_desc": [
            "- `best.pt`    — checkpoint complet du meilleur epoch (EXAM AUC_mal)",
            "- `last.pt`    — checkpoint de reprise (--resume)",
            "- `logs.json`  — métriques par epoch (vue + exam)",
            "- `args.json`  — hyperparams bruts (source de vérité machine)",
            "- `train.log`  — stdout de l'entraînement",
            "- `split_patients.json` — PID train/abstention/val (à réutiliser tel quel)",
        ],
        "argv": vars(args),
    }
    write_args_json(run_dir, meta)
    write_run_readme(run_dir, meta)

    # --- modele ---
    params = default_parameters(percent_t=args.percent_t, gpu_number=0, dropout=args.dropout)
    model = build_gmic(args.weights, params, device)
    if args.freeze_backbone:
        freeze_backbone(model)
    if args.freeze_global:
        freeze_global(model)
    if args.backbone_dropout > 0:
        add_backbone_dropout(model, args.backbone_dropout)

    optimizer = torch.optim.Adam(
        param_groups(model, args.lr_backbone, args.lr_head),
        weight_decay=args.weight_decay,
    )
    warmup_iters = max(1, args.warmup_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_iters))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_iters])
    scaler = GradScaler("cuda", enabled=use_amp)

    logs = []
    best_auc = -1.0
    best_epoch = None
    early_stopped = False
    epochs_no_improve = 0
    start_epoch = 0
    ckpt_last = os.path.join(args.out, "last.pt")
    ckpt_best = os.path.join(args.out, "best.pt")
    logs_path = os.path.join(args.out, "logs.json")

    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        best_auc = ck.get("best_auc", -1.0)
        logs = ck.get("logs", [])
        start_epoch = ck["epoch"] + 1
        print(f"[resume] reprise a l'epoch {start_epoch} (best_auc={best_auc:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_backbone:
            set_backbone_eval(model)
        if args.freeze_global:
            set_global_eval(model)
        t0 = time.time()
        agg = {"loss": 0.0, "bce_fusion": 0.0, "bce_global": 0.0,
               "bce_local": 0.0, "l1_saliency": 0.0}
        nb = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float()
            if gpu_aug is not None:
                x = gpu_aug(x)          # augmentation sur GPU (train only)
            x = gpu_standardize(x)
            yt = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                yf, yg, yl, sal = model(x)
            # BCE en fp32, hors autocast
            total, parts = compute_loss(yf, yg, yl, sal, yt, args.beta)
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k in agg:
                agg[k] += parts[k]
            nb += 1
        scheduler.step()
        for k in agg:
            agg[k] /= max(nb, 1)

        metrics = evaluate(model, val_loader, val_samples, device, args.beta)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        entry = {"epoch": epoch, "time_s": round(dt, 1), "lr_backbone": lr_now,
                 **{f"train_{k}": round(v, 5) for k, v in agg.items()},
                 **{k: (round(v, 5) if v == v else None) for k, v in metrics.items()}}
        logs.append(entry)
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"[epoch {epoch:03d}] {dt:5.0f}s | train_loss={agg['loss']:.4f} "
              f"| val_loss={metrics['val_loss']:.4f} "
              f"| vue AUC_mal={metrics['auc_malignant']:.4f} "
              f"| EXAM AUC_mal={metrics['exam_auc_malignant']:.4f} "
              f"(ben={metrics['exam_auc_benign']:.4f}, n={metrics['n_exams']})")
        print(f"           [tetes] EXAM AUC_mal  fusion={metrics['exam_auc_malignant']:.4f} "
              f"global={metrics['exam_auc_global_mal']:.4f} local={metrics['exam_auc_local_mal']:.4f} "
              f"| train BCE  fus={agg['bce_fusion']:.3f} glo={agg['bce_global']:.3f} loc={agg['bce_local']:.3f}")

        save_ckpt(ckpt_last, epoch, model, optimizer, scheduler, scaler, best_auc, logs)
        # Selection + early stopping sur l'AUC_malignant NIVEAU EXAMEN (par sein)
        auc_m = metrics["exam_auc_malignant"]
        if auc_m == auc_m and auc_m > best_auc + args.min_delta:  # not NaN et amelioration nette
            best_auc = auc_m
            best_epoch = epoch
            epochs_no_improve = 0
            save_ckpt(ckpt_best, epoch, model, optimizer, scheduler, scaler, best_auc, logs)
            print(f"           -> nouveau best EXAM AUC_mal={best_auc:.4f} (sauve dans best.pt)")
        else:
            epochs_no_improve += 1
            print(f"           pas d'amelioration ({epochs_no_improve}/{args.patience}) "
                  f"| best EXAM AUC_mal={best_auc:.4f}")
            if epochs_no_improve >= args.patience:
                early_stopped = True
                print(f"[early stopping] arret a l'epoch {epoch} : "
                      f"{args.patience} epochs sans amelioration. Best EXAM AUC_mal={best_auc:.4f}")
                break

    # --- README final (resultats) ---
    total_s = time.time() - t_run0
    meta.update({
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_auc": best_auc if best_auc > 0 else None,
        "best_epoch": best_epoch,
        "epochs_ran": len(logs),
        "early_stopped": early_stopped,
        "total_time_human": format_duration(total_s),
        "total_time_s": round(total_s, 1),
    })
    write_args_json(run_dir, meta)
    write_run_readme(run_dir, meta)
    print(f"Termine. Best EXAM AUC malignant (par sein) = {best_auc:.4f} "
          f"(epoch {best_epoch}) | run documente dans {run_dir}")


if __name__ == "__main__":
    main()
