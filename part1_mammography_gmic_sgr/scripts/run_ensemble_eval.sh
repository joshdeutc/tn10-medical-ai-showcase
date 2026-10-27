#!/usr/bin/env bash
# Eval d'ensemble GMIC, A LANCER QUAND LES 4 FINETUNINGS SONT FINIS.
# Retrouve tout seul les best.pt horodates de m2..m5.
set -u
cd ~/breast_cancer
RUNS=fine_tuning/checkpoints/runs/cancer_malignant_gmic

latest() { ls -dt "$1"/*/best.pt 2>/dev/null | head -1; }

M1=fine_tuning/checkpoints/gmic_ft_exam/best.pt          # model_1 deja finetune
M2=$(latest $RUNS/gmic_nyu-m2_gpu-aug)
M3=$(latest $RUNS/gmic_nyu-m3_gpu-aug)
M4=$(latest $RUNS/gmic_nyu-m4_gpu-aug)
M5=$(latest $RUNS/gmic_nyu-m5_gpu-aug)
REG=fine_tuning/checkpoints/gmic_ft_reg/best.pt
AUG=fine_tuning/checkpoints/gmic_ft_aug/best.pt
FZG=$(latest $RUNS/gmic_freeze-global_gpu-aug)
FT=$(latest $RUNS/gmic_full-ft_gpu-aug_fulltrain)

echo "Checkpoints trouves :"
for v in M1 M2 M3 M4 M5 REG AUG FZG FT; do echo "  $v = ${!v:-<MANQUANT>}"; done

for v in M2 M3 M4 M5; do
  if [ -z "${!v:-}" ]; then echo "ERREUR: $v manquant -> les finetunings ne sont pas tous finis."; exit 1; fi
done

echo
echo "############ 1) LES 5 VRAIS MODELES NYU (percent_t propres) ############"
./.venv/bin/python _ensemble_eval.py \
  --ckpts $M1 $M2 $M3 $M4 $M5 \
  --names m1 m2 m3 m4 m5 \
  --percent-ts 0.02 0.03 0.03 0.05 0.10

echo
echo "############ 2) LES 5 NYU + VARIANTES DIVERSES (m1) ############"
./.venv/bin/python _ensemble_eval.py \
  --ckpts $M1 $M2 $M3 $M4 $M5 $REG $AUG $FZG $FT \
  --names m1 m2 m3 m4 m5 reg aug freezeglobal fulltrain \
  --percent-ts 0.02 0.03 0.03 0.05 0.10 0.02 0.02 0.02 0.02
