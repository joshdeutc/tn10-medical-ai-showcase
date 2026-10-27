"""
Validation des donnees d'entree avant le pipeline GMIC
------------------------------------------------------
Verifie que les images et le CSV sont conformes aux attentes
du modele GMIC. A lancer AVANT le pipeline pour detecter
les problemes tot.

Usage :
  python utils/validate_input.py --input-dir data/
  python utils/validate_input.py --input-dir data/ --csv data/train.csv --strict
"""

import os
import sys
import csv
import argparse
import random

import cv2
import numpy as np


# ── Seuils de validation ────────────────────────────────────────────────────

MIN_IMAGE_SIZE = 700       # pixels minimum (hauteur et largeur)
MIN_PIXEL_RANGE = 50       # ecart min-max minimum (image pas toute noire/blanche)
VERY_LARGE_DIM = 5000      # avertissement si une dimension depasse ce seuil
EXPECTED_CSV_COLS = {"patient_id", "image_id", "laterality", "view", "cancer"}
VALID_LATERALITIES = {"L", "R"}
VALID_VIEWS = {"CC", "MLO"}
MAX_SAMPLE = 50            # nombre d'images a verifier (pour ne pas tout lire)


# ── Resultats ────────────────────────────────────────────────────────────────

class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def ok(self, msg):
        self.info.append(msg)

    @property
    def passed(self):
        return len(self.errors) == 0

    def print_report(self):
        print("\n" + "=" * 60)
        print("RAPPORT DE VALIDATION")
        print("=" * 60)

        if self.info:
            for msg in self.info:
                print(f"  [OK]    {msg}")

        if self.warnings:
            print()
            for msg in self.warnings:
                print(f"  [WARN]  {msg}")

        if self.errors:
            print()
            for msg in self.errors:
                print(f"  [FAIL]  {msg}")

        print()
        if self.passed:
            print(f"Resultat : VALIDE ({len(self.info)} checks OK, {len(self.warnings)} warnings)")
        else:
            print(f"Resultat : ECHEC ({len(self.errors)} erreurs, {len(self.warnings)} warnings)")
            print("Corrigez les erreurs avant de lancer le pipeline.")
        print()


# ── Checks CSV ───────────────────────────────────────────────────────────────

def check_csv(csv_path: str, result: ValidationResult):
    """Verifie le format et le contenu du CSV."""

    if not os.path.exists(csv_path):
        result.error(f"CSV introuvable : {csv_path}")
        return None
    if os.path.isdir(csv_path):
        result.error(f"Le chemin CSV pointe vers un dossier, pas un fichier : {csv_path}")
        return None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])

    # Colonnes requises
    missing = EXPECTED_CSV_COLS - columns
    if missing:
        result.error(
            f"Colonnes manquantes dans le CSV : {missing}. "
            f"Colonnes attendues : {EXPECTED_CSV_COLS}"
        )
        return None
    result.ok(f"CSV : colonnes requises presentes ({', '.join(sorted(EXPECTED_CSV_COLS))})")

    # Lire le contenu
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Lateralite
    bad_lat = set(df["laterality"].unique()) - VALID_LATERALITIES
    if bad_lat:
        result.error(f"Lateralites invalides : {bad_lat}. Attendu : {VALID_LATERALITIES}")
    else:
        result.ok("CSV : lateralites valides (L, R)")

    # Vues
    bad_view = set(df["view"].unique()) - VALID_VIEWS
    if bad_view:
        result.error(f"Vues invalides : {bad_view}. Attendu : {VALID_VIEWS}")
    else:
        result.ok("CSV : vues valides (CC, MLO)")

    # Label cancer
    cancer_vals = set(df["cancer"].unique())
    if not cancer_vals.issubset({0, 1}):
        result.error(f"Valeurs 'cancer' invalides : {cancer_vals}. Attendu : 0 ou 1")
    else:
        n_pos = int((df["cancer"] == 1).sum())
        n_neg = int((df["cancer"] == 0).sum())
        result.ok(f"CSV : {len(df)} lignes, {df['patient_id'].nunique()} patients "
                  f"({n_pos} cancer, {n_neg} sains)")
        if n_pos == 0:
            result.warn("Aucun patient cancer=1 dans le CSV")
        if n_neg == 0:
            result.warn("Aucun patient cancer=0 dans le CSV")

    return df


# ── Checks images ────────────────────────────────────────────────────────────

def find_images_dir(input_dir: str) -> str:
    """Trouve le dossier contenant les images (train_images/ ou directement)."""
    train_images = os.path.join(input_dir, "train_images")
    if os.path.isdir(train_images):
        return train_images
    return input_dir


def check_image(path: str, result: ValidationResult, strict: bool = False) -> bool:
    """
    Verifie qu'une image est une mammographie valide.
    Retourne True si OK, False sinon.
    """
    basename = os.path.basename(path)

    # Lecture
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        result.error(f"Image illisible : {path}")
        return False

    # Couleur ?
    if len(img.shape) == 3 and img.shape[2] == 3:
        result.error(
            f"Image en COULEUR (RGB) : {path} — "
            f"Les mammographies sont en niveaux de gris. "
            f"Ceci n'est probablement pas une mammographie."
        )
        return False
    if len(img.shape) == 3 and img.shape[2] == 4:
        result.error(
            f"Image RGBA (avec transparence) : {path} — "
            f"Ce n'est pas une mammographie."
        )
        return False

    h, w = img.shape[:2]

    # Taille minimum (warning seulement — le pipeline peut continuer)
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        result.warn(
            f"Image petite : {path} ({w}x{h}). "
            f"Les mammographies font typiquement >2000px. "
            f"Seuil recommande : {MIN_IMAGE_SIZE}px."
        )

    # Avertissement si image tres grande
    if max(h, w) > VERY_LARGE_DIM:
        result.warn(
            f"Image tres grande : {path} ({w}x{h}). "
            f"Une dimension depasse {VERY_LARGE_DIM}px — "
            f"cela peut ralentir le pipeline."
        )

    # Toute noire ou toute blanche ?
    pmin, pmax = int(img.min()), int(img.max())
    pixel_range = pmax - pmin
    if pixel_range < MIN_PIXEL_RANGE:
        result.error(
            f"Image presque uniforme : {path} "
            f"(min={pmin}, max={pmax}, ecart={pixel_range}). "
            f"L'image est probablement corrompue ou vide."
        )
        return False

    # Ratio d'aspect suspect (image carree ou tres large = probablement pas une mammo)
    ratio = h / w if w > 0 else 0
    if strict and (ratio < 0.5 or ratio > 5.0):
        result.warn(
            f"Ratio hauteur/largeur suspect : {path} ({w}x{h}, ratio={ratio:.2f}). "
            f"Les mammographies ont typiquement un ratio entre 0.8 et 3.0."
        )

    return True


def check_images(images_dir: str, result: ValidationResult, df=None,
                 strict: bool = False):
    """Verifie un echantillon d'images."""

    if not os.path.isdir(images_dir):
        result.error(f"Dossier d'images introuvable : {images_dir}")
        return

    SUPPORTED_FORMATS = (".png", ".dcm", ".dcm.zip")
    UNSUPPORTED_FORMATS = (".jpg", ".jpeg", ".webp", ".avif", ".bmp",
                           ".tiff", ".tif", ".gif", ".heic", ".heif")

    all_images = []
    unsupported = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            flo = f.lower()
            if flo.endswith(SUPPORTED_FORMATS):
                all_images.append(os.path.join(root, f))
            elif flo.endswith(UNSUPPORTED_FORMATS):
                unsupported.append(os.path.join(root, f))

    # Signaler les formats non supportes
    if unsupported:
        exts = set(os.path.splitext(f)[1].lower() for f in unsupported)
        examples = [os.path.basename(f) for f in unsupported[:3]]
        result.error(
            f"{len(unsupported)} fichier(s) avec un format non supporte "
            f"({', '.join(sorted(exts))}) : {examples}. "
            f"Le pipeline accepte uniquement PNG (mammographies converties) "
            f"ou DICOM (.dcm / .dcm.zip). "
            f"Les photos JPEG/WebP/AVIF ne sont pas des mammographies."
        )

    if not all_images:
        if not unsupported:
            result.error(f"Aucune image trouvee dans {images_dir}")
        return

    # Detecter le format
    has_dcm = any(f.endswith(".dcm") or f.endswith(".dcm.zip") for f in all_images)
    has_png = any(f.endswith(".png") for f in all_images)
    if has_dcm and has_png:
        result.warn("Melange de DICOM et PNG dans le meme dossier")
    fmt = "DICOM" if has_dcm else "PNG"
    result.ok(f"Images : {len(all_images)} fichiers {fmt} trouves dans {images_dir}")

    # Verifier la coherence CSV <-> fichiers
    if df is not None:
        csv_ids = set()
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            iid = str(row["image_id"])
            csv_ids.add(f"{pid}/{iid}")

        disk_ids = set()
        for img_path in all_images:
            parts = img_path.replace(images_dir, "").strip("/").split("/")
            if len(parts) >= 2:
                pid = parts[0]
                iid = os.path.splitext(parts[1])[0].replace(".dcm", "")
                disk_ids.add(f"{pid}/{iid}")

        in_csv_not_disk = csv_ids - disk_ids
        in_disk_not_csv = disk_ids - csv_ids

        if in_csv_not_disk:
            n = len(in_csv_not_disk)
            examples = list(in_csv_not_disk)[:3]
            result.warn(
                f"{n} images dans le CSV mais absentes du disque. "
                f"Exemples : {examples}"
            )
        if in_disk_not_csv:
            n = len(in_disk_not_csv)
            result.warn(f"{n} images sur disque mais absentes du CSV (seront ignorees)")

        matched = csv_ids & disk_ids
        result.ok(f"Coherence CSV/disque : {len(matched)} images trouvees sur les "
                  f"{len(csv_ids)} du CSV")

    # Echantillonner des images PNG pour verification pixel
    png_images = [f for f in all_images if f.endswith(".png")]
    if png_images:
        sample = random.sample(png_images, min(MAX_SAMPLE, len(png_images)))
        n_ok, n_fail = 0, 0
        for path in sample:
            if check_image(path, result, strict=strict):
                n_ok += 1
            else:
                n_fail += 1
        if n_fail == 0:
            result.ok(f"Echantillon de {len(sample)} images : toutes valides")
    elif has_dcm:
        result.ok("Images DICOM detectees — la validation pixel se fera apres conversion")


# ── Checks GMIC ─────────────────────────────────────────────────────────────

def check_gmic(result: ValidationResult):
    """Verifie que le modele GMIC est en place."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    gmic_dir = os.path.join(project_dir, "GMIC")

    if not os.path.isdir(gmic_dir):
        result.error(f"Dossier GMIC introuvable : {gmic_dir}")
        return

    models_dir = os.path.join(gmic_dir, "models")
    expected_models = [f"sample_model_{i}.p" for i in range(1, 6)]
    missing = [m for m in expected_models if not os.path.exists(os.path.join(models_dir, m))]

    if missing:
        result.error(
            f"Modeles GMIC manquants : {missing}. "
            f"Placez-les dans {models_dir}/"
        )
    else:
        result.ok("GMIC : 5 modeles presents")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validation des donnees d'entree pour le pipeline GMIC"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Dossier contenant les images")
    parser.add_argument("--csv", default=None,
                        help="Chemin vers le CSV (defaut: auto-detection)")
    parser.add_argument("--strict", action="store_true",
                        help="Mode strict : les warnings deviennent des erreurs")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    result = ValidationResult()

    print("\n" + "=" * 60)
    print("VALIDATION DES DONNEES D'ENTREE")
    print("=" * 60)

    # 1. GMIC
    print("\n[1/3] Verification du modele GMIC...")
    check_gmic(result)

    # 2. CSV
    print("[2/3] Verification du CSV...")
    csv_path = args.csv
    if csv_path is None:
        for candidate in ["train_subset_test.csv", "train.csv"]:
            p = os.path.join(input_dir, candidate)
            if os.path.exists(p) and os.path.isfile(p):
                csv_path = p
                break
    if csv_path:
        df = check_csv(csv_path, result)
    else:
        result.error(f"Aucun CSV trouve dans {input_dir}. "
                     f"Utilisez --csv pour specifier le chemin.")
        df = None

    # 3. Images
    print("[3/3] Verification des images...")
    images_dir = find_images_dir(input_dir)
    check_images(images_dir, result, df=df, strict=args.strict)

    # Rapport
    result.print_report()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
