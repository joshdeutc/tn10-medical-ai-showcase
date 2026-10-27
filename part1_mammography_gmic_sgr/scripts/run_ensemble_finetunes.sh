#!/usr/bin/env bash
# Finetune les 4 modeles NYU manquants (model_1 deja fait) pour l'ensemble GMIC.
# Chacun avec SON percent_t (PERCENT_T_DICT). Meme split 2/3 + gpu-augment que le
# reste -> val identique -> ensemblable. Sequentiel (1 GPU).
set -u
cd ~/breast_cancer
for spec in "2 0.03" "3 0.03" "4 0.05" "5 0.10"; do
  set -- $spec
  M=$1; PT=$2
  echo "========================================================"
  echo "=== FINETUNE sample_model_${M}  (percent_t=${PT}) ==="
  echo "========================================================"
  ./.venv/bin/python -m fine_tuning.train_gmic \
    --weights GMIC/models/sample_model_${M}.p \
    --percent-t ${PT} \
    --gpu-augment --batch-size 16 --num-workers 6 \
    --epochs 40 --patience 6 \
    --model-tag gmic_nyu-m${M}_gpu-aug
  echo "=== model_${M} TERMINE ($(date '+%H:%M:%S')) ==="
done
echo "######## LES 4 FINETUNINGS SONT TERMINES ########"
