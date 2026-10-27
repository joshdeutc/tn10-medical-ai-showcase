"""
Inférence GMIC sur un dataset préprocessé par preprocess.py.

Utilise ScratchGMIC (gmic_from_scratch.py) — pas de best_center requis.
Chargement direct des PNG avec imageio + normalisation z-score.

Sauvegarde incrémentale : chaque image est écrite dans le CSV dès qu'elle
est traitée. Le script peut être interrompu et relancé : les images déjà
présentes dans le CSV sont skippées.

Sortie structurée : les prédictions vont dans un dossier de run horodaté
    inference/runs/<dataset>/<model_tag>/<timestamp>/
        predictions.csv, meta.json, README.md
(même logique que fine_tuning/checkpoints/runs/). Utiliser --run-dir pour
écrire dans un dossier précis (et reprendre un run interrompu).

Usage :
    uv run python scripts/inference.py --output-dir data/preprocess_image/rsna_output
    uv run python scripts/inference.py --output-dir data/preprocess_image/rsna_output --model-index 1
    uv run python scripts/inference.py --output-dir data/preprocess_image/sample --run-dir inference/runs/sample/gmic-nyu-sample1/manuel
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))           # pour importer gmic_from_scratch
sys.path.insert(0, str(_PROJECT_ROOT))   # pour importer fine_tuning.run_metadata

from gmic_from_scratch import ScratchGMIC, load_nyu_weights  # noqa: E402
from fine_tuning.run_metadata import make_run_dir, get_git_commit  # noqa: E402

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
FIELDNAMES = ["image_index", "malignant_pred", "benign_pred", "malignant_label"]

PERCENT_T = {"1": 0.02, "2": 0.03, "3": 0.03, "4": 0.05, "5": 0.1}


def load_done(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["image_index"])
    return done


def build_model(model_index: str, device: torch.device) -> ScratchGMIC:
    model = ScratchGMIC(
        K=6,
        crop_shape=(256, 256),
        cam_size=(46, 30),
        percent_t=PERCENT_T[model_index],
        num_classes=2,
        device_type="gpu" if device.type == "cuda" else "cpu",
        gpu_number=device.index or 0,
    ).to(device).eval()

    ckpt = _PROJECT_ROOT / "GMIC" / "models" / f"sample_model_{model_index}.p"
    load_nyu_weights(model, str(ckpt), device=device)
    return model


def run(exam_list, model, image_dir: str, csv_path: str, device: torch.device, done: set) -> int:
    write_header = not os.path.exists(csv_path)
    written = 0

    with open(csv_path, "a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with torch.no_grad():
            for datum in tqdm.tqdm(exam_list, desc="Exams"):
                for view in VIEWS:
                    images_for_view = datum.get(view, [])
                    if not images_for_view:
                        continue
                    short_path = images_for_view[0]

                    if short_path in done:
                        continue

                    img_file = os.path.join(image_dir, short_path + ".png")
                    if not os.path.exists(img_file):
                        tqdm.tqdm.write(f"[WARN] introuvable : {img_file}")
                        continue

                    try:
                        img = imageio.imread(img_file).astype(np.float32)
                        if img.shape != (2944, 1920):
                            img = cv2.resize(img, (1920, 2944), interpolation=cv2.INTER_AREA)
                        img = (img - img.mean()) / max(img.std(), 1e-5)
                        x = torch.tensor(img[None, None]).to(device)
                        y = model(x)
                    except Exception as e:
                        tqdm.tqdm.write(f"[WARN] erreur sur {short_path}: {type(e).__name__}: {e}")
                        continue

                    benign_pred = float(y[0, 0])
                    malignant_pred = float(y[0, 1])

                    cancer = datum["cancer_label"]
                    if view.startswith("L-"):
                        malignant_label = float(cancer.get("left_malignant", 0))
                    else:
                        malignant_label = float(cancer.get("right_malignant", 0))

                    writer.writerow({
                        "image_index": short_path,
                        "malignant_pred": f"{malignant_pred:.4f}",
                        "benign_pred": f"{benign_pred:.4f}",
                        "malignant_label": f"{malignant_label:.4f}",
                    })
                    done.add(short_path)
                    written += 1

    return written


def _rel(path) -> str:
    """Chemin relatif au projet si possible, sinon absolu (pour les meta)."""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_run_metadata(run_dir: Path, *, dataset, model_index, model_tag, ckpt,
                       input_dir, image_dir, n_exams, n_predictions, device) -> None:
    """Écrit `meta.json` (machine) + `README.md` (humain) dans le dossier de run."""
    meta = {
        "run_kind": "gmic_inference",
        "dataset": dataset,
        "model_tag": model_tag,
        "model_index": model_index,
        "checkpoint": _rel(ckpt),
        "input_preprocess_dir": _rel(input_dir),
        "image_dir": _rel(image_dir),
        "n_exams": n_exams,
        "n_predictions": n_predictions,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(_PROJECT_ROOT),
        "script": "scripts/inference.py",
        "columns": {
            "image_index": "identifiant court de l'image (dossier/nom)",
            "malignant_pred": "score GMIC de malignité (classe positive)",
            "benign_pred": "score GMIC de bénignité",
            "malignant_label": "vérité terrain (1 = cancer, 0 = sain)",
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    readme = [
        f"# Run d'inférence GMIC — {run_dir.name}",
        "",
        "## Modèle",
        f"- **Architecture** : ScratchGMIC (poids NYU pré-entraînés, non re-entraînés)",
        f"- **Checkpoint** : `{meta['checkpoint']}` (model-index {model_index})",
        "",
        "## Données",
        f"- **Dataset** : `{dataset}`",
        f"- **Dossier préprocessé (entrée)** : `{meta['input_preprocess_dir']}`",
        f"- **Examens** : {n_exams}  |  **Prédictions** : {n_predictions}",
        "",
        "## Exécution",
        f"- **Date** : {meta['timestamp']}",
        f"- **Device** : {device}",
        f"- **Git commit** : `{meta['git_commit']}`",
        "",
        "## Fichiers",
        "- `predictions.csv` — image_index, malignant_pred, benign_pred, malignant_label",
        "- `meta.json`       — métadonnées machine (source de vérité)",
        "",
        "Évaluation / abstention :",
        f"    uv run python scripts/eval_gmic.py --predictions {_rel(run_dir / 'predictions.csv')}",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(readme))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Dataset préprocessé par preprocess.py (ENTRÉE : data.pkl + cropped_images/)")
    parser.add_argument("--model-index", default="1", choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--dataset", default=None,
                        help="Nom du dataset pour structurer le run (défaut : nom du dossier --output-dir)")
    parser.add_argument("--runs-root", default=None,
                        help="Racine des runs d'inférence (défaut : <projet>/inference/runs)")
    parser.add_argument("--run-dir", default=None,
                        help="Écrit dans CE dossier précis (créé si absent, reprend si predictions.csv existe). "
                             "Sinon : dossier horodaté auto sous runs-root/<dataset>/<model_tag>/<timestamp>/")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.output_dir)
    pkl_path = os.path.join(input_dir, "data.pkl")
    image_dir = os.path.join(input_dir, "cropped_images")

    if not os.path.exists(pkl_path):
        print(f"Erreur : data.pkl introuvable dans {input_dir}")
        sys.exit(1)

    # Résolution du dossier de run (sortie structurée)
    model_tag = f"gmic-nyu-sample{args.model_index}"
    dataset = args.dataset or os.path.basename(input_dir.rstrip("/")) or "dataset"
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_root = Path(args.runs_root).resolve() if args.runs_root else _PROJECT_ROOT / "inference" / "runs"
        run_dir = make_run_dir(runs_root, dataset, model_tag)
    csv_path = os.path.join(run_dir, "predictions.csv")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_number}")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Run    : {_rel(run_dir)}")

    with open(pkl_path, "rb") as f:
        exam_list = pickle.load(f)
    print(f"Exams : {len(exam_list)}")

    done = load_done(csv_path)
    print(f"Déjà traités : {len(done)} images")

    model = build_model(args.model_index, device)
    ckpt = _PROJECT_ROOT / "GMIC" / "models" / f"sample_model_{args.model_index}.p"

    written = run(exam_list, model, image_dir, csv_path, device, done)
    n_predictions = len(load_done(csv_path))

    write_run_metadata(
        run_dir, dataset=dataset, model_index=args.model_index, model_tag=model_tag,
        ckpt=ckpt, input_dir=input_dir, image_dir=image_dir,
        n_exams=len(exam_list), n_predictions=n_predictions, device=device,
    )

    print(f"\nTerminé — {written} nouvelles prédictions → {csv_path}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
