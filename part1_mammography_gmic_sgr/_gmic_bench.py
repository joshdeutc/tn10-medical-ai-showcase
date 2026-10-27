#!/usr/bin/env python
"""Mini-benchmark de vitesse/VRAM pour le finetuning GMIC sur la VM (L40S 46GB).

Replique exactement un step d'entrainement (forward AMP + loss 3-tetes + backward
+ grad-clip + step) et chronometre N steps apres un warmup, en mesurant la VRAM pic.
Sert a estimer la duree d'une epoch (1/3 train) et a valider le batch-size.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "GMIC"))
sys.path.insert(0, ROOT)

from fine_tuning.dataset_gmic import make_loaders, INPUT_SIZE  # noqa
from fine_tuning.gmic_finetune import build_gmic, default_parameters, param_groups  # noqa
from fine_tuning.train_gmic import compute_loss  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p")
    ap.add_argument("--percent-t", type=float, default=0.02)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | batch={args.batch_size} "
          f"| workers={args.num_workers}")

    train_loader, _, train_samples, _ = make_loaders(
        pkl_path="data/preprocess_image/data.pkl",
        image_dir="data/preprocess_image/cropped_images",
        batch_size=args.batch_size, num_workers=args.num_workers,
        augment=True, input_size=INPUT_SIZE,
    )
    n_train_views = len(train_samples)
    steps_per_epoch = n_train_views // args.batch_size

    params = default_parameters(percent_t=args.percent_t, gpu_number=0)
    model = build_gmic(args.weights, params, device)
    model.train()
    optimizer = torch.optim.Adam(param_groups(model, 1e-5, 1e-4), weight_decay=1e-5)
    scaler = GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    it = iter(train_loader)
    t_start = None
    done = 0
    target = args.warmup + args.steps
    while done < target:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        yt = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            yf, yg, yl, sal = model(x)
        total, _ = compute_loss(yf, yg, yl, sal, yt, 1e-4)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        done += 1
        if done == args.warmup:
            torch.cuda.synchronize()
            t_start = time.time()
        if done % 10 == 0:
            print(f"  step {done}/{target}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    sps = args.steps / elapsed
    peak = torch.cuda.max_memory_allocated() / 1e9

    print("=" * 56)
    print(f"vues train (1/3)      : {n_train_views}")
    print(f"steps/epoch (batch {args.batch_size:>2}) : {steps_per_epoch}")
    print(f"sec/step              : {elapsed/args.steps:.3f}")
    print(f"steps/s               : {sps:.2f}")
    print(f"VRAM pic              : {peak:.1f} GB / 46 GB")
    epoch_s = steps_per_epoch / sps
    print(f"=> ~{epoch_s/60:.1f} min/epoch (train seul)")
    print(f"=> ~{epoch_s*30/3600:.1f} h pour 30 epochs (hors val/early-stop)")
    print("=" * 56)


if __name__ == "__main__":
    main()
