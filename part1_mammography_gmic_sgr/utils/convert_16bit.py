#!/usr/bin/env python
"""
Conversion 16 bits du subset train+val : DICOM -> PNG uint16 croppe 2944x1920.

Pourquoi : les PNG de cropped_images/ sont en uint8 (perte 12->8 bits au
preprocessing, cf utils/preprocess.py etape 5). On regenere les MEMES crops en
uint16 a VALEURS NATIVES 12 bits (pas de rescale 65535 : le z-score au
chargement annule toute echelle lineaire, et l'octet de poids fort quasi vide
compresse bien mieux en PNG).

Garanties de geometrie IDENTIQUE aux PNG 8 bits existants :
  - fenetre de crop : reprise de cropped_exam_list.pkl quand elle existe ;
    sinon recalculee par crop_img_from_largest_connected (deterministe, masque
    `img > 0` insensible a l'echelle d'intensite -> meme fenetre qu'a l'epoque).
  - resize : meme regle que preprocess._resize_one (INTER_AREA si plus grand,
    INTER_LINEAR sinon).
  - flip horizontal des vues R (orientation chest-wall-a-gauche).

Tout est fait EN MEMOIRE par image (decode -> crop -> resize -> flip -> save) :
aucun intermediaire plein format sur disque (piege de saturation de 2026-06).

Perimetre : les PID des sets demandes du split_patients.json (defaut :
train+val du split canonique 2/3-1/6-1/6 seed 42 ; l'abstention 1/6 n'est PAS
convertie, reservee au futur modele d'abstention).

Usage (sur la VM, dans tmux) :
    ./.venv/bin/python utils/convert_16bit.py --num-processes 8
    # reprise : relancer tel quel (skip si le PNG de sortie existe)
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GMIC"))

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
GMIC_H, GMIC_W = 2944, 1920
# parametres du crop GMIC, identiques au CLI crop_mammogram.py du pipeline
CROP_ITERATIONS = 100
CROP_BUFFER = 50
CROP_RIGHTMOST_RATIO = 1.0 / 3


def _pid_of(exam):
    for v in VIEWS:
        fs = exam.get(v) or []
        if fs:
            return fs[0].split("/")[0]
    return None


def _crop_mode(view):
    # image_orientation(horizontal_flip="NO", side) du pipeline : L->right, R->left
    return "left" if view.startswith("R") else "right"


def load_jobs(split_json, sets, data_pkl, window_pkls, raw_dir, out_dir):
    """Construit la liste des (pid, iid, view, window|None, dcm_path, out_path)."""
    sp = json.load(open(split_json))
    need = set()
    for s in sets:
        need |= set(sp["pids"][s])

    with open(data_pkl, "rb") as f:
        exams = pickle.load(f)

    # pid -> exam avec window_location (les pkl plus recents ecrasent)
    win_exam = {}
    for path in window_pkls:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            for e in pickle.load(f):
                p = _pid_of(e)
                if p and "window_location" in e:
                    win_exam[p] = e

    jobs, with_win, without_win = [], 0, 0
    for exam in exams:
        pid = _pid_of(exam)
        if pid is None or pid not in need:
            continue
        we = win_exam.get(pid)
        for v in VIEWS:
            for j, fs in enumerate(exam.get(v) or []):
                iid = fs.split("/")[1]
                win = None
                if we is not None:
                    wl = we.get("window_location", {}).get(v) or []
                    if j < len(wl):
                        win = tuple(int(x) for x in wl[j])
                if win is None:
                    without_win += 1
                else:
                    with_win += 1
                jobs.append((pid, iid, v, win,
                             os.path.join(raw_dir, pid, iid + ".dcm"),
                             os.path.join(out_dir, pid, iid + ".png")))
    print(f"jobs: {len(jobs)} vues sur {len(need)} PID "
          f"| window stockee: {with_win} | a recalculer: {without_win}")
    return jobs


def convert_one(job):
    """Worker : decode DICOM natif -> crop -> resize -> flip -> PNG uint16."""
    pid, iid, view, win, dcm_path, out_path = job
    if os.path.exists(out_path):
        return ("skip", 0)
    if not os.path.exists(dcm_path):
        return ("miss", 0)
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr            # invert, echelle native conservee
        arr = arr.astype(np.uint16)          # valeurs natives (12 bits -> <=4095)

        if win is None:
            # fenetre recalculee = identique a l'epoque (masque img>0,
            # insensible a l'echelle d'intensite)
            from src.cropping.crop_mammogram import crop_img_from_largest_connected
            win = crop_img_from_largest_connected(
                arr, _crop_mode(view), True,
                CROP_ITERATIONS, CROP_BUFFER, CROP_RIGHTMOST_RATIO)[0]

        top, bottom, left, right = (int(x) for x in win)
        img = arr[top:bottom, left:right]

        h, w = img.shape
        if (h, w) != (GMIC_H, GMIC_W):
            interp = cv2.INTER_AREA if (h > GMIC_H or w > GMIC_W) else cv2.INTER_LINEAR
            img = cv2.resize(img, (GMIC_W, GMIC_H), interpolation=interp)

        if view.startswith("R"):
            img = cv2.flip(img, 1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not cv2.imwrite(out_path, img):
            return ("err", 0)
        return ("ok", int(img.max()))
    except Exception as e:
        print(f"  ERREUR {pid}/{iid} ({view}) : {type(e).__name__}: {e}", flush=True)
        return ("err", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json",
                    default="fine_tuning/checkpoints/gmic_ft_exam/split_patients.json",
                    help="split canonique patient-level (2/3-1/6-1/6 seed 42)")
    ap.add_argument("--sets", default="train,val",
                    help="sets du split a convertir (l'abstention reste vierge)")
    ap.add_argument("--data-pkl", default="data/preprocess_image/data.pkl")
    ap.add_argument("--window-pkls", nargs="*", default=[
        "data/preprocess_image/cropped_exam_list.pkl",
        "data/preprocess_image_new/cropped_exam_list.pkl",
    ])
    ap.add_argument("--raw-dir", default="data/rsna/train_images")
    ap.add_argument("--out", default="data/preprocess_image/cropped_16bit")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=40.0,
                    help="garde-fou : arret si le disque passe sous ce seuil")
    ap.add_argument("--limit", type=int, default=0, help="debug : ne traiter que N vues")
    args = ap.parse_args()

    jobs = load_jobs(args.split_json, args.sets.split(","), args.data_pkl,
                     args.window_pkls, args.raw_dir, args.out)
    if args.limit:
        jobs = jobs[:args.limit]

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    maxvals = []
    with Pool(args.num_processes) as pool:
        for i, (status, mx) in enumerate(pool.imap_unordered(convert_one, jobs,
                                                             chunksize=8), 1):
            stats[status] += 1
            if status == "ok":
                maxvals.append(mx)
            if i % 500 == 0 or i == len(jobs):
                free_gb = shutil.disk_usage("/").free / 1e9
                rate = i / max(time.time() - t0, 1e-9)
                eta_min = (len(jobs) - i) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(jobs)} {stats} | {rate:.1f} img/s "
                      f"| ETA {eta_min:.0f} min | free {free_gb:.0f} GB", flush=True)
                if free_gb < args.min_free_gb:
                    print(f"STOP : disque sous {args.min_free_gb} GB libres", flush=True)
                    sys.exit(1)

    print(f"\nFINI {stats} en {(time.time()-t0)/60:.1f} min")
    if maxvals:
        mv = np.array(maxvals)
        print(f"max pixel par image : min={mv.min()} mediane={int(np.median(mv))} "
              f"max={mv.max()} | >4095 : {(mv > 4095).sum()} images "
              f"| >32767 (deborde int16) : {(mv > 32767).sum()} images")


if __name__ == "__main__":
    main()
