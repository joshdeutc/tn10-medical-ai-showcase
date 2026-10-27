"""
Chargement flexible d'images mammographiques.

Accepte un fichier unique ou un dossier (récursif).

Usage CLI :
    python utils/load_images.py image.png
    python utils/load_images.py data/preprocess_image/sample/cropped_images/
    python utils/load_images.py data/preprocess_image/sample/cropped_images/ --max 10

Fonctions importables :
    from utils.load_images import collect_images, load_image, load_all
"""

import argparse
import glob
import os

import imageio
import numpy as np


def collect_images(path: str) -> list[str]:
    """Retourne la liste des PNG trouvés dans path (fichier ou dossier)."""
    p = os.path.abspath(path)

    if os.path.isfile(p):
        if not p.lower().endswith(".png"):
            raise ValueError(f"Le fichier n'est pas un PNG : {p}")
        return [p]

    if os.path.isdir(p):
        found = sorted(glob.glob(os.path.join(p, "**", "*.png"), recursive=True))
        if not found:
            raise FileNotFoundError(f"Aucun PNG trouvé dans : {p}")
        return found

    raise FileNotFoundError(f"Chemin introuvable : {p}")


def load_image(path: str) -> np.ndarray:
    """Charge un PNG et applique la normalisation z-score (float32)."""
    img = imageio.imread(path).astype(np.float32)
    img = (img - img.mean()) / max(img.std(), 1e-5)
    return img


def load_all(path: str, max_images: int | None = None) -> tuple[list[str], list[np.ndarray]]:
    """
    Charge toutes les images d'un fichier ou dossier.

    Returns:
        paths  : liste des chemins chargés
        images : liste de tableaux numpy normalisés (float32)
    """
    paths = collect_images(path)
    if max_images is not None:
        paths = paths[:max_images]

    images = []
    for p in paths:
        images.append(load_image(p))

    return paths, images


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Charge des images PNG (fichier ou dossier).")
    parser.add_argument("path", help="Chemin vers un fichier PNG ou un dossier")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="Nombre maximum d'images à charger")
    args = parser.parse_args()

    paths, images = load_all(args.path, max_images=args.max)

    print(f"{len(images)} image(s) chargée(s)")
    for p, img in zip(paths, images):
        print(f"  {p}  →  shape={img.shape}  mean={img.mean():.3f}  std={img.std():.3f}")
