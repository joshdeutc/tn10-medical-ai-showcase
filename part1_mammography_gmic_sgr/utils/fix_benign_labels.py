#!/usr/bin/env python
"""
Corrige le label `benign` dans data.pkl.

Le preprocessing (utils/preprocess.py) met benign = NOT malignant (tout sein
cancer==0 -> benign=1), ce qui donne une prevalence ~50% medicalement fausse et
ignore la colonne `biopsy` de RSNA.

Le vrai benign RSNA = sein biopsie mais sans cancer : biopsy==1 & cancer==0.

Ce script regenere `cancer_label` de chaque exam a partir de train.csv, SANS
toucher aux images (pas de re-crop). Backup unique -> data.pkl.prebenignfix.

A LANCER APRES la fin du preprocessing (le merge reecrit data.pkl a chaque batch).

Usage :
    ./.venv/bin/python utils/fix_benign_labels.py \
        --pkl data/preprocess_image/data.pkl \
        --csv data/rsna/train.csv
"""
import argparse
import os
import pickle
import shutil
from collections import Counter

import pandas as pd

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]


def exam_pid(exam):
    for v in VIEWS:
        fs = exam.get(v) or []
        if fs:
            return fs[0].split("/")[0]
    return None


def build_side_labels(csv_path):
    """Retourne deux dicts {(patient_id_str, 'L'|'R'): 0/1} pour malignant et benign."""
    df = pd.read_csv(csv_path)
    df["cancer"] = df["cancer"].astype(int)
    df["biopsy"] = df["biopsy"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)

    # malignant = il existe au moins une image cancer==1 pour ce (patient, cote)
    mal = df.groupby(["patient_id", "laterality"])["cancer"].max()

    # benign = il existe au moins une image biopsy==1 & cancer==0 pour ce (patient, cote)
    df["benign_row"] = ((df["biopsy"] == 1) & (df["cancer"] == 0)).astype(int)
    ben = df.groupby(["patient_id", "laterality"])["benign_row"].max()

    return mal.to_dict(), ben.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/preprocess_image/data.pkl")
    ap.add_argument("--csv", default="data/rsna/train.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="affiche la nouvelle distribution sans ecrire")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        exams = pickle.load(f)
    print(f"Exams charges : {len(exams)}  ({args.pkl})")

    mal_d, ben_d = build_side_labels(args.csv)

    before = Counter()
    after = Counter()
    missing = 0

    for exam in exams:
        pid = exam_pid(exam)
        cl = exam["cancer_label"]
        # distribution avant
        for k in ("left_benign", "right_benign", "left_malignant", "right_malignant"):
            before[(k, int(cl.get(k, 0)))] += 1

        if pid is None:
            missing += 1
            continue

        lm = int(mal_d.get((pid, "L"), 0))
        rm = int(mal_d.get((pid, "R"), 0))
        lb = int(ben_d.get((pid, "L"), 0))
        rb = int(ben_d.get((pid, "R"), 0))

        cl["left_malignant"] = lm
        cl["right_malignant"] = rm
        cl["malignant"] = int(lm or rm)
        cl["left_benign"] = lb
        cl["right_benign"] = rb
        cl["benign"] = int(lb or rb)
        cl["unknown"] = 0

        for k, val in (("left_benign", lb), ("right_benign", rb),
                       ("left_malignant", lm), ("right_malignant", rm)):
            after[(k, val)] += 1

    def show(counter, title):
        print(f"  {title}")
        for k in ("left_benign", "right_benign", "left_malignant", "right_malignant"):
            pos = counter[(k, 1)]
            neg = counter[(k, 0)]
            tot = pos + neg
            pct = 100.0 * pos / tot if tot else 0.0
            print(f"    {k:16s} =1 : {pos:6d}  ({pct:5.1f}%)")

    print("Distribution AVANT :")
    show(before, "avant")
    print("Distribution APRES (biopsy & ~cancer) :")
    show(after, "apres")
    if missing:
        print(f"[ATTENTION] {missing} exams sans patient_id identifiable (inchanges)")

    if args.dry_run:
        print("--dry-run : aucun fichier ecrit.")
        return

    backup = args.pkl + ".prebenignfix"
    if not os.path.exists(backup):
        shutil.copy2(args.pkl, backup)
        print(f"Backup -> {backup}")
    else:
        print(f"Backup deja present ({backup}), conserve.")

    with open(args.pkl, "wb") as f:
        pickle.dump(exams, f)
    print(f"data.pkl reecrit avec labels benign corriges : {args.pkl}")


if __name__ == "__main__":
    main()
