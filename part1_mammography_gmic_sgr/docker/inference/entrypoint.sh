#!/usr/bin/env bash
# Point d'entrée TOUT-EN-UN de l'image GMIC : un menu de sous-commandes qui
# expose toutes les fonctions du projet. Sans sous-commande connue, la 1re
# valeur est exécutée telle quelle (pass-through : `bash`, `python ...`, etc.).
#
# Variables d'environnement (valeurs par défaut pour la démo) :
#   INPUT_DIR    images brutes à préprocesser   (défaut : data/raw/sample)
#   OUTPUT_DIR   dossier de sortie              (défaut : data/preprocess_image/sample)
#   MODEL_INDEX  modèle NYU 1..5                (défaut : 1)
#   NUM_PROC     processus parallèles du crop   (défaut : 2)
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/raw/sample}"
OUTPUT_DIR="${OUTPUT_DIR:-data/preprocess_image/sample}"
MODEL_INDEX="${MODEL_INDEX:-1}"
NUM_PROC="${NUM_PROC:-2}"
# Dossier de run d'inférence (structuré) utilisé par demo/pipeline.
INFER_RUN_DIR="${INFER_RUN_DIR:-inference/runs/sample/gmic-nyu-sample${MODEL_INDEX}/demo}"

usage() {
    cat <<EOF
GMIC — image autonome tout-en-un. Usage : docker run --rm <image> <commande> [args]

Commandes :
  demo                 (défaut) preprocess + inférence sur le sample embarqué
  pipeline             demo + évaluation (preprocess -> inférence -> éval)
  preprocess [args]    utils/preprocess.py        (crop + resize)
  infer [args]         scripts/inference.py         (--output-dir par défaut: $OUTPUT_DIR)
  eval [args]          scripts/eval_gmic.py         (métriques sur predictions.csv)
  train [args]         fine_tuning/train_resnet.py  (entraînement ResNet)
  test [args]          pytest des tests unitaires
  smoke                auto-test complet de l'image (preprocess->inférence->éval + pytest)
  help                 cette aide
  <autre>              exécuté tel quel (ex. bash, python scripts/xxx.py --help)

Variables d'env : INPUT_DIR, OUTPUT_DIR, MODEL_INDEX, NUM_PROC
EOF
}

run_demo() {
    echo "==============================================================="
    echo " GMIC — démo : preprocess + inférence (image autonome)"
    echo " Images : ${INPUT_DIR}   Sortie : ${OUTPUT_DIR}   Modèle : ${MODEL_INDEX}"
    echo "==============================================================="
    echo
    echo "### ÉTAPE 1/2 — Preprocessing (crop + resize 2944×1920)"
    python utils/preprocess.py --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" --num-processes "${NUM_PROC}"
    echo
    echo "### ÉTAPE 2/2 — Inférence GMIC"
    python scripts/inference.py --output-dir "${OUTPUT_DIR}" --model-index "${MODEL_INDEX}" \
        --run-dir "${INFER_RUN_DIR}"
    echo
    echo "=== Prédictions : ${INFER_RUN_DIR}/predictions.csv ==="
    cat "${INFER_RUN_DIR}/predictions.csv"
}

cmd="${1:-demo}"
case "$cmd" in
    demo)
        run_demo
        ;;
    pipeline)
        run_demo
        echo
        echo "### ÉTAPE 3/3 — Évaluation"
        python scripts/eval_gmic.py --predictions "${INFER_RUN_DIR}/predictions.csv"
        ;;
    preprocess)
        shift; exec python utils/preprocess.py "$@"
        ;;
    infer|inference)
        shift
        if [ "$#" -eq 0 ]; then
            exec python scripts/inference.py --output-dir "${OUTPUT_DIR}" --model-index "${MODEL_INDEX}" \
                --run-dir "${INFER_RUN_DIR}"
        fi
        exec python scripts/inference.py "$@"
        ;;
    eval)
        shift; exec python scripts/eval_gmic.py "$@"
        ;;
    train)
        shift; exec python fine_tuning/train_resnet.py "$@"
        ;;
    test)
        shift; exec python -m pytest utils/test_validate_input.py -v "$@"
        ;;
    smoke)
        exec smoke_test.sh
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        exec "$@"
        ;;
esac
