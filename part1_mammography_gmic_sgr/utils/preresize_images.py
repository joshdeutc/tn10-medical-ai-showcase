"""
Pré-resize sur disque des images croppées vers une résolution cible
(par défaut 512×512, format uint8 PNG).

Objectif : éliminer le coût de resize à chaque epoch dans les DataLoaders
PyTorch. Les images croppées d'origine font ~2944×1920 (≈6 MB/image) ; après
pré-resize à 512×512 uint8 on descend à ~120 KB/image, soit ~30× moins d'I/O.

One-shot : à relancer chaque fois que cropped_images/ change (nouveau run de
preprocess.py). Les images déjà présentes dans la destination sont skippées,
donc c'est incrémental si tu ajoutes de nouveaux patients.

Usage :
    python utils/preresize_images.py
    python utils/preresize_images.py --size 384
    python utils/preresize_images.py --src <...> --dst <...>
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_src = project_root / "data" / "preprocess_image" / "rsna_output" / "cropped_images"
    default_dst = project_root / "data" / "preprocess_image" / "rsna_output" / "cropped_512"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=default_src,
                    help=f"Dossier source (defaut: {default_src})")
    ap.add_argument("--dst", type=Path, default=default_dst,
                    help=f"Dossier destination (defaut: {default_dst})")
    ap.add_argument("--size", type=int, default=512,
                    help="Resolution carree cible (defaut: 512)")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"Erreur : dossier source introuvable : {args.src}", file=sys.stderr)
        sys.exit(1)

    pngs = sorted(args.src.rglob("*.png"))
    if not pngs:
        print(f"Erreur : aucun PNG trouve sous {args.src}", file=sys.stderr)
        sys.exit(1)

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print(f"Taille : {args.size}x{args.size}")
    print(f"Fichiers a traiter : {len(pngs)}")

    resized, skipped, failed = 0, 0, 0
    for p in tqdm(pngs, desc="preresize", unit="img"):
        out = args.dst / p.relative_to(args.src)
        if out.exists():
            skipped += 1
            continue
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                failed += 1
                continue
            img_r = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img_r)
            resized += 1
        except Exception as e:
            print(f"  Erreur {p} : {e}", file=sys.stderr)
            failed += 1

    print(f"\nResizees : {resized} | Deja presentes : {skipped} | Echecs : {failed}")
    print(f"Total sur disque : {args.dst}")


if __name__ == "__main__":
    main()
