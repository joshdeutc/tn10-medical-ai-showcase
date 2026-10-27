"""
Prétraitement GMIC — Étapes 1 à 6
-----------------------------------
Prépare les images (DICOM ou PNG) pour l'inférence GMIC :

  1. Détection du format d'entrée (DICOM ou PNG)
  2. Conversion DICOM -> PNG 16-bit (si nécessaire)
  3. Construction du PKL au format GMIC
  4. Recadrage (crop_mammogram.py)
  5. Redimensionnement à 2944×1920 + normalisation uint8
  6. Flip horizontal des vues droites (R-CC, R-MLO)

GMIC a été entraîné avec tous les seins orientés vers la gauche. Les vues
droites (R-CC, R-MLO) sont naturellement orientées à droite : on les retourne
horizontalement UNE FOIS ici, lors du preprocessing, pour que toutes les images
stockées soient déjà dans l'orientation attendue par le modèle.
→ inference.py et fine_tuning/dataset.py n'ont plus besoin de gérer ce flip.

Note : la normalisation mean/std (z-score) reste appliquée à la volée lors
du chargement car elle produit des float32 négatifs — incompatibles avec le
format PNG uint8 utilisé pour le stockage.

Le dossier de sortie contiendra :
  <output-dir>/cropped_images/         <- images prêtes pour l'inférence
  <output-dir>/data.pkl                <- PKL final (à passer à inference.py)
  <output-dir>/exam_list_before_cropping.pkl
  <output-dir>/cropped_exam_list.pkl

Usage :
  python utils/preprocess.py --input-dir data/raw/sample --output-dir data/preprocess_image/sample
  python utils/preprocess.py --input-dir data/raw/extract_dataset3 --output-dir data/preprocess_image/extract_dataset3
  python utils/preprocess.py --input-dir data/raw/sample --output-dir data/preprocess_image/sample --force-crop
"""

import os
import sys
import csv
import glob
import pickle
import shutil
import zipfile
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GMIC_DIR = os.path.join(PROJECT_DIR, "GMIC")

GMIC_H, GMIC_W = 2944, 1920

sys.path.insert(0, GMIC_DIR)


# ── Détection automatique du format ──────────────────────────────────────────

def detect_format(input_dir: str) -> str:
    search_dir = input_dir
    train_images = os.path.join(input_dir, "train_images")
    if os.path.isdir(train_images):
        search_dir = train_images

    for patient_dir in sorted(os.listdir(search_dir))[:20]:
        patient_path = os.path.join(search_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        for f in os.listdir(patient_path):
            if f.endswith(".dcm.zip") or f.endswith(".dcm"):
                return "dicom"
            if f.endswith(".png"):
                return "png"

    return "unknown"


# ── Étape 1 : Conversion DICOM -> PNG 16-bit ─────────────────────────────────

def _convert_dcm_one(task):
    """Worker (module-level, picklable pour le Pool) : convertit 1 DICOM -> PNG
    16-bit. Regles IDENTIQUES a l'historique : MONOCHROME1 inverse, normalisation
    [0,65535], skip si le PNG existe deja. Retourne un statut
    ('ok' / 'skip' / 'missing' / 'err:<pid>/<iid>:<msg>')."""
    pid, iid, raw_dir, png_dir = task
    out_path = os.path.join(png_dir, pid, f"{iid}.png")
    if os.path.exists(out_path):
        return "skip"

    zip_path = os.path.join(raw_dir, pid, f"{iid}.dcm.zip")
    dcm_path = os.path.join(raw_dir, pid, f"{iid}.dcm")
    try:
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                dcm_name = next(n for n in z.namelist() if n.endswith(".dcm"))
                with z.open(dcm_name) as f_dcm:
                    ds = pydicom.dcmread(f_dcm)
        elif os.path.exists(dcm_path):
            ds = pydicom.dcmread(dcm_path)
        else:
            return "missing"

        arr = ds.pixel_array.astype(np.float32)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr
        arr_max = arr.max()
        if arr_max > 0:
            arr = (arr / arr_max * 65535).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)

        os.makedirs(os.path.join(png_dir, pid), exist_ok=True)
        cv2.imwrite(out_path, arr)
        return "ok"
    except Exception as e:
        return f"err:{pid}/{iid}:{e}"


def convert_dcm_to_png(raw_dir: str, png_dir: str, csv_path: str, num_processes: int = 4):
    """Conversion DICOM -> PNG 16-bit PARALLELE (Pool de `num_processes` procs).

    Absorbe l'ancien `convert_dcm_parallel.py` (HOME) : c'etait le seul etage encore
    mono-thread du pipeline. Logique de conversion inchangee ; skip-existant conserve.
    """
    print("\n" + "=" * 60)
    print("ETAPE 1 : Conversion DICOM -> PNG 16-bit")
    print("=" * 60)

    image_ids = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            image_ids.add((row["patient_id"], row["image_id"]))
    tasks = [(pid, iid, raw_dir, png_dir) for pid, iid in sorted(image_ids)]

    converted, skipped, failed = 0, 0, 0
    nproc = max(1, num_processes)
    with Pool(nproc) as pool:
        for status in tqdm(pool.imap_unordered(_convert_dcm_one, tasks, chunksize=8),
                           total=len(tasks), desc=f"Conversion ({nproc} procs)"):
            if status == "ok":
                converted += 1
            elif status == "skip":
                skipped += 1
            elif status == "missing":
                failed += 1
            else:  # err:...
                failed += 1
                print(f"  Erreur {status[4:]}")

    print(f"\nConvertis : {converted} | Deja OK : {skipped} | Echecs : {failed}")
    return converted + skipped


# ── Étape 2 : Construction du PKL GMIC ───────────────────────────────────────

def build_exam_pkl(csv_path: str, png_dir: str, pkl_path: str) -> list:
    print("\n" + "=" * 60)
    print("ETAPE 2 : Construction du PKL GMIC")
    print("=" * 60)

    patients: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            if pid not in patients:
                patients[pid] = {
                    "horizontal_flip": "NO",
                    "L-CC": [], "L-MLO": [], "R-CC": [], "R-MLO": [],
                    "L-CC_benign_seg": [], "L-CC_malignant_seg": [],
                    "L-MLO_benign_seg": [], "L-MLO_malignant_seg": [],
                    "R-CC_benign_seg": [], "R-CC_malignant_seg": [],
                    "R-MLO_benign_seg": [], "R-MLO_malignant_seg": [],
                    "cancer_label": {
                        "benign": 0, "left_benign": 0, "right_benign": 0,
                        "malignant": 0, "left_malignant": 0, "right_malignant": 0,
                        "unknown": 0,
                    },
                }

            iid = row["image_id"]
            lat = row["laterality"]
            view = row["view"]
            cancer = int(row["cancer"])
            biopsy = int(row["biopsy"])
            view_key = f"{lat}-{view}"

            png_path = os.path.join(png_dir, pid, f"{iid}.png")
            if not os.path.exists(png_path):
                continue

            if view_key in patients[pid]:
                patients[pid][view_key].append(f"{pid}/{iid}")

            # Labels RSNA :
            #   malignant = cancer biopsié (cancer == 1)
            #   benign    = biopsié SANS cancer (biopsy == 1 & cancer == 0)
            # Un sein jamais biopsié (biopsy == 0, cancer == 0) n'est NI benign
            # NI malignant -> les deux flags restent 0 (et pas "unknown", car
            # l'absence de biopsie en dépistage = négatif radiologique, pas un
            # cas indéterminé). Surtout : NE PAS faire benign = NOT malignant,
            # ce qui étiquetterait ~98 % des seins sains comme benign.
            side = "left" if lat == "L" else "right"
            if cancer == 1:
                patients[pid]["cancer_label"]["malignant"] = 1
                patients[pid]["cancer_label"][f"{side}_malignant"] = 1
            if biopsy == 1 and cancer == 0:
                patients[pid]["cancer_label"]["benign"] = 1
                patients[pid]["cancer_label"][f"{side}_benign"] = 1

    VIEWS = ("L-CC", "L-MLO", "R-CC", "R-MLO")
    exam_list = [e for e in patients.values() if any(e[v] for v in VIEWS)]
    dropped = len(patients) - len(exam_list)

    with open(pkl_path, "wb") as f:
        pickle.dump(exam_list, f)

    mal_count = sum(1 for e in exam_list if e["cancer_label"]["malignant"])
    ben_count = sum(1 for e in exam_list if e["cancer_label"]["benign"])
    print(f"Examens : {len(exam_list)} total | {mal_count} malignant | "
          f"{ben_count} benign (biopsy & ~cancer)")
    if dropped > 0:
        print(f"Patients fantomes ignores (aucune image locale) : {dropped}")
    print(f"PKL sauvegarde : {pkl_path}")
    return exam_list


# ── Étape 3 : Crop ───────────────────────────────────────────────────────────

def run_crop(png_dir: str, cropped_dir: str, pkl_raw: str, pkl_cropped: str, num_processes: int = 4):
    print("\n" + "=" * 60)
    print("ETAPE 3 : Recadrage (crop_mammogram)")
    print("=" * 60)

    if os.path.exists(cropped_dir):
        print(f"Suppression de l'ancien dossier {cropped_dir}...")
        shutil.rmtree(cropped_dir)

    with open(pkl_raw, "rb") as f:
        exam_list_raw = pickle.load(f)
    total_images = sum(
        len(exam[v]) for exam in exam_list_raw
        for v in ["L-CC", "R-CC", "L-MLO", "R-MLO"]
    )
    print(f"Images a cropper : {total_images}")

    cmd = (
        f"cd {GMIC_DIR} && PYTHONPATH={GMIC_DIR}:$PYTHONPATH {sys.executable} "
        f"src/cropping/crop_mammogram.py "
        f"--input-data-folder {png_dir} "
        f"--output-data-folder {cropped_dir} "
        f"--exam-list-path {pkl_raw} "
        f"--cropped-exam-list-path {pkl_cropped} "
        f"--num-processes {num_processes}"
    )

    import subprocess
    import threading
    import time as _time

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _progress():
        while proc.poll() is None:
            done = len(glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True))
            print(f"\r  Croppes : {done}/{total_images}", end="", flush=True)
            _time.sleep(1)
        done = len(glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True))
        print(f"\r  Croppes : {done}/{total_images}", flush=True)

    t = threading.Thread(target=_progress, daemon=True)
    t.start()
    proc.wait()
    t.join()

    if proc.returncode == 0:
        print("Crop termine")
    else:
        err = proc.stderr.read().decode()
        print(f"Crop echoue (code {proc.returncode})")
        if err:
            print(err[-500:])
        sys.exit(1)


# ── Étape 4 : Resize 2944×1920 + normalisation ───────────────────────────────

def _resize_one(args):
    """Worker parallele : resize/normalise une image en place.
    Retourne (status, sfp_ou_None, scale_ou_None). Logique identique a la
    boucle sequentielle d'origine."""
    path, cropped_dir = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return ("none", None, None)
    h, w = img.shape[:2]
    if h == GMIC_H and w == GMIC_W:
        if img.max() > 255 or img.dtype != np.uint8:
            cv2.imwrite(path, _normalize_uint8(img))
            return ("skipnorm", None, None)
        return ("skip", None, None)
    rel = os.path.relpath(path, cropped_dir)
    sfp = rel[:-4] if rel.endswith(".png") else rel
    interp = cv2.INTER_AREA if (h > GMIC_H or w > GMIC_W) else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (GMIC_W, GMIC_H), interpolation=interp)
    norm = False
    if img_resized.max() > 255 or img_resized.dtype != np.uint8:
        img_resized = _normalize_uint8(img_resized)
        norm = True
    cv2.imwrite(path, img_resized)
    return ("resizednorm" if norm else "resized", sfp, (GMIC_H / h, GMIC_W / w))


def resize_all(cropped_dir: str, pkl_cropped: str, output_dir: str = None):
    print("\n" + "=" * 60)
    print(f"ETAPE 4 : Redimensionnement a {GMIC_H}x{GMIC_W} + normalisation [0, 255]")
    print("=" * 60)
    if output_dir is None:
        output_dir = cropped_dir

    import src.utilities.pickling as pickling
    import src.utilities.data_handling as data_handling

    exam_list = pickling.unpickle_from_file(pkl_cropped)

    # Scan tous les PNG du dossier — pas seulement ceux dans le pkl,
    # car data.pkl peut référencer plus d'images que cropped_exam_list.pkl.
    all_pngs = glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True)

    scale_map = {}
    resized, skipped, normalized = 0, 0, 0

    from multiprocessing import Pool
    nproc = min(8, os.cpu_count() or 4)
    with Pool(nproc) as pool:
        results = pool.map(
            _resize_one,
            [(p, cropped_dir) for p in all_pngs],
            chunksize=16,
        )

    for status, sfp, scale in results:
        if status == "none":
            continue
        if status in ("skip", "skipnorm"):
            skipped += 1
            if status == "skipnorm":
                normalized += 1
            continue
        # resized / resizednorm
        scale_map[sfp] = scale
        resized += 1
        if status == "resizednorm":
            normalized += 1

    print(f"Redimensionnees : {resized} | Deja OK : {skipped} | Normalisees [0,255] : {normalized}")

    VIEWS_LIST = ["L-CC", "R-CC", "L-MLO", "R-MLO"]
    updated = 0
    for exam in exam_list:
        for view in VIEWS_LIST:
            if view not in exam or not exam[view]:
                continue
            for j in range(len(exam[view])):
                sfp = exam[view][j]
                rp = exam["rightmost_points"][view][j]
                bp = exam["bottommost_points"][view][j]
                wl = exam["window_location"][view][j]

                needs_fix = False
                if int(rp[1]) > GMIC_W:
                    needs_fix = True
                if int(bp[0]) > GMIC_H:
                    needs_fix = True
                if sfp in scale_map:
                    needs_fix = True

                if not needs_fix:
                    continue

                if sfp in scale_map:
                    sh, sw = scale_map[sfp]
                else:
                    # rightmost_points[1] ≈ largeur crop, bottommost_points[0] ≈ hauteur crop
                    orig_h = float(bp[0]) if bp[0] > 0 else GMIC_H
                    orig_w = float(rp[1]) if rp[1] > 0 else GMIC_W
                    sh = GMIC_H / orig_h
                    sw = GMIC_W / orig_w

                (ry1, ry2), rx = rp
                exam["rightmost_points"][view][j] = (
                    (int(round(float(ry1) * sh)), int(round(float(ry2) * sh))),
                    int(round(float(rx) * sw)),
                )

                by, (bx1, bx2) = bp
                exam["bottommost_points"][view][j] = (
                    int(round(float(by) * sh)),
                    (int(round(float(bx1) * sw)), int(round(float(bx2) * sw))),
                )

                # window_location = (y_top, y_bottom, x_left, x_right) dans l'image
                # originale pre-crop — ces coordonnees ne changent pas avec le resize.
                # On ne modifie pas window_location ici.

                updated += 1

    if updated > 0:
        pickling.pickle_to_file(pkl_cropped, exam_list)
        print(f"PKL mis a jour : {updated} images avec coordonnees recalculees")

    open(os.path.join(output_dir, _RESIZE_MARKER), "w").close()


def _normalize_uint8(img):
    img_f = img.astype(np.float32)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max > img_min:
        return ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return np.zeros_like(img_f, dtype=np.uint8)


# ── Étape 5 : Flip horizontal des vues droites ───────────────────────────────

_RESIZE_MARKER = ".resize_done"
_FLIP_MARKER = ".right_views_flipped"
_VIEWS_RIGHT = {"R-CC", "R-MLO"}


def _flip_one(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    cv2.imwrite(path, cv2.flip(img, 1))  # 1 = axe vertical = flip horizontal
    return 1


def apply_right_view_flip(cropped_dir: str, pkl_path: str, output_dir: str):
    """
    Retourne horizontalement toutes les images R-CC et R-MLO dans cropped_dir.

    GMIC a été entraîné avec les seins orientés à gauche. Les vues droites sont
    naturellement orientées à droite : ce flip les aligne avec les vues gauches
    et avec les attentes du modèle.

    Idempotent : un marqueur .right_views_flipped est créé pour éviter
    d'appliquer le flip deux fois lors d'une ré-exécution.
    """
    print("\n" + "=" * 60)
    print("ETAPE 5 : Flip horizontal des vues droites (R-CC, R-MLO)")
    print("=" * 60)

    with open(pkl_path, "rb") as f:
        exam_list = pickle.load(f)

    paths = []
    for exam in exam_list:
        for view in _VIEWS_RIGHT:
            for sfp in exam.get(view, []):
                path = os.path.join(cropped_dir, sfp + ".png")
                if os.path.exists(path):
                    paths.append(path)

    from multiprocessing import Pool
    nproc = min(8, os.cpu_count() or 4)
    with Pool(nproc) as pool:
        flipped = sum(pool.map(_flip_one, paths, chunksize=16))

    open(os.path.join(output_dir, _FLIP_MARKER), "w").close()
    print(f"Images retournees : {flipped}")


def is_flip_done(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, _FLIP_MARKER))


# ── Étape 6 : get_optimal_centers ────────────────────────────────────────────

def run_optimal_centers(cropped_dir: str, pkl_cropped: str, pkl_final: str):
    print("\n" + "=" * 60)
    print("ETAPE 6 : Calcul des centres optimaux")
    print("=" * 60)

    cmd = (
        f"cd {GMIC_DIR} && PYTHONPATH={GMIC_DIR}:$PYTHONPATH {sys.executable} "
        f"src/optimal_centers/get_optimal_centers.py "
        f"--cropped-exam-list-path {pkl_cropped} "
        f"--data-prefix {cropped_dir} "
        f"--output-exam-list-path {pkl_final} "
        f"--num-processes 4"
    )
    ret = os.system(cmd)
    if ret == 0:
        print("get_optimal_centers termine")
    else:
        print(f"get_optimal_centers echoue (code {ret})")
        sys.exit(1)


# ── Détection automatique des étapes déjà effectuées ─────────────────────────

def _count_pngs(directory: str) -> int:
    return len(glob.glob(os.path.join(directory, "**", "*.png"), recursive=True))


def is_crop_done(cropped_dir: str, pkl_cropped: str) -> bool:
    if not os.path.exists(pkl_cropped) or not os.path.exists(cropped_dir):
        return False
    return _count_pngs(cropped_dir) > 0


def is_resize_done(cropped_dir: str, pkl_cropped: str, output_dir: str = None) -> bool:
    if output_dir is None:
        output_dir = cropped_dir
    return os.path.exists(os.path.join(output_dir, _RESIZE_MARKER))


def copy_pkl_as_final(pkl_cropped: str, pkl_final: str):
    """Copie cropped_exam_list.pkl en data.pkl — aucun best_center nécessaire."""
    shutil.copy(pkl_cropped, pkl_final)
    print(f"data.pkl genere depuis cropped_exam_list.pkl")


def is_final_done(pkl_final: str) -> bool:
    return os.path.exists(pkl_final)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prétraitement GMIC — Prépare les images pour l'inférence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python utils/preprocess.py --input-dir data/raw/sample --output-dir data/preprocess_image/sample
  python utils/preprocess.py --input-dir data/raw/extract_dataset3 --output-dir data/preprocess_image/extract_dataset3
  python utils/preprocess.py --input-dir data/raw/sample --output-dir data/preprocess_image/sample --force-crop
        """,
    )
    parser.add_argument("--input-dir", default=None,
                        help="Dossier contenant les images (DICOM ou PNG). "
                             "Optionnel si --force-resize sans --force-crop.")
    parser.add_argument("--csv", default=None,
                        help="Chemin vers le CSV. Par defaut : <input-dir>/train_subset_test.csv ou train.csv")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier de sortie. Par defaut : <projet>/data/preprocess_image/")
    parser.add_argument("--format", choices=["dicom", "png", "auto"], default="auto",
                        help="Format des images d'entree (defaut: auto-detection)")
    parser.add_argument("--force-crop", action="store_true",
                        help="Forcer le crop meme s'il semble deja fait")
    parser.add_argument("--force-resize", action="store_true",
                        help="Forcer le resize meme s'il semble deja fait")
    parser.add_argument("--num-processes", type=int, default=4,
                        help="Nombre de processes pour crop_mammogram (defaut: 4)")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "data", "preprocess_image")
    output_dir = os.path.abspath(output_dir)

    png_dir = os.path.join(output_dir, "png_images")
    cropped_dir = os.path.join(output_dir, "cropped_images")
    pkl_raw = os.path.join(output_dir, "exam_list_before_cropping.pkl")
    pkl_cropped = os.path.join(output_dir, "cropped_exam_list.pkl")
    pkl_final = os.path.join(output_dir, "data.pkl")

    os.makedirs(output_dir, exist_ok=True)

    # --force-resize sans --force-crop : les données brutes ne sont pas nécessaires
    # si le crop est déjà fait (cropped_exam_list.pkl présent).
    resize_only = args.force_resize and not args.force_crop and is_crop_done(cropped_dir, pkl_cropped)

    if not resize_only:
        if args.input_dir is None:
            print("Erreur : --input-dir est requis sauf si --force-resize avec crop deja effectue.")
            sys.exit(1)

        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            print(f"Erreur : dossier introuvable : {input_dir}")
            sys.exit(1)

        csv_path = args.csv
        if csv_path is None:
            csv_path = os.path.join(input_dir, "train_subset_test.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(input_dir, "train.csv")
        if not os.path.exists(csv_path):
            print(f"Erreur : CSV introuvable : {csv_path}")
            sys.exit(1)

        fmt = args.format
        if fmt == "auto":
            fmt = detect_format(input_dir)
            if fmt == "unknown":
                print("Erreur : impossible de detecter le format des images.")
                print("Specifiez --format dicom ou --format png")
                sys.exit(1)
        print(f"Format detecte : {fmt.upper()}")

        if fmt == "dicom":
            raw_dir = os.path.join(input_dir, "train_images")
            if not os.path.isdir(raw_dir):
                raw_dir = input_dir
            os.makedirs(png_dir, exist_ok=True)
            convert_dcm_to_png(raw_dir, png_dir, csv_path, num_processes=args.num_processes)
            source_png_dir = png_dir
        else:
            train_images = os.path.join(input_dir, "train_images")
            source_png_dir = train_images if os.path.isdir(train_images) else input_dir

        with open(os.path.join(output_dir, "source_dir.txt"), "w") as _f:
            _f.write(source_png_dir)

        build_exam_pkl(csv_path, source_png_dir, pkl_raw)

    if args.force_crop:
        # Nouveau crop → les images sont fraîches et non retournées : réinitialiser le marqueur
        marker = os.path.join(output_dir, _FLIP_MARKER)
        if os.path.exists(marker):
            os.remove(marker)
        # data.pkl doit aussi être régénéré — sinon il reste l'ancien avec une liste de patients périmée
        if os.path.exists(pkl_final):
            os.remove(pkl_final)
        run_crop(source_png_dir, cropped_dir, pkl_raw, pkl_cropped, num_processes=args.num_processes)
    elif is_crop_done(cropped_dir, pkl_cropped):
        n = _count_pngs(cropped_dir)
        print(f"\n[AUTO] Crop : {n} images deja croppees -> SKIP")
    else:
        run_crop(source_png_dir, cropped_dir, pkl_raw, pkl_cropped, num_processes=args.num_processes)

    if args.force_resize:
        # Resize en place : l'orientation des images ne change pas.
        # Ne pas toucher au marqueur de flip pour eviter un double flip.
        resize_all(cropped_dir, pkl_cropped, output_dir)
    elif is_resize_done(cropped_dir, pkl_cropped, output_dir):
        print(f"[AUTO] Resize : deja effectue -> SKIP")
    else:
        resize_all(cropped_dir, pkl_cropped, output_dir)

    if is_flip_done(output_dir):
        print(f"[AUTO] Flip vues droites : deja applique -> SKIP")
    else:
        apply_right_view_flip(cropped_dir, pkl_cropped, output_dir)

    if is_final_done(pkl_final):
        print(f"[AUTO] PKL final : {pkl_final} deja present -> SKIP")
    else:
        copy_pkl_as_final(pkl_cropped, pkl_final)

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINE")
    print(f"  Images cropppees : {_count_pngs(cropped_dir)}")
    print(f"  PKL final        : {pkl_final}")
    print(f"\nLancer l'inference avec :")
    print(f"  python scripts/inference.py --output-dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
