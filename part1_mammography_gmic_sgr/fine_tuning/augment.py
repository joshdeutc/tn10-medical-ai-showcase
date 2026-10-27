"""
Augmentation et normalisation GPU pour le finetuning GMIC.

Extrait de `train_gmic.py` (15 juin 2026) pour separer les responsabilites :
ce module ne contient que les transformations appliquees sur GPU au batch
d'images (augmentation kornia train-only + z-score par image). Importe par
`train_gmic.py` et par les scripts de bench (`_speedbench`, `_vram_probe`) /
d'evaluation (`_ensemble_*`, `_abstain_*`).

`train_gmic` re-exporte `GPUAugment` et `gpu_standardize` (via son import) pour
ne casser aucun `from fine_tuning.train_gmic import ...` existant.
"""
import torch
import torch.nn as nn

from fine_tuning.dataset_gmic import INPUT_SIZE


class GPUAugment(nn.Module):
    """Augmentation kornia sur GPU -- memes transformations que dataset._augment
    (CPU) : affine (rotation +/-15, scale 0.9-1.1, translation +/-max_shift px,
    fond noir) + jitter gamma (p=0.5, 0.8-1.25). PAS de hflip (vues deja
    normalisees chest-wall-a-gauche). Train only (a ne pas appeler en eval).

    Entree : float (B,1,H,W), echelle libre (uint8 [0,255] OU uint16 natif
    12 bits) : on normalise par le MAX PAR IMAGE avant gamma -> pour les uint8
    minmax-stretches (max~255) c'est identique a l'historique /255, et pour le
    16 bits natif le gamma garde la meme force relative. Le z-score en aval
    annule toute echelle lineaire -> seul le gamma non-lineaire survit.
    """

    def __init__(self, input_size=INPUT_SIZE, max_shift=(48, 32)):
        super().__init__()
        import kornia.augmentation as K
        # kornia translate = fractions (horizontale/W, verticale/H)
        tx = max_shift[1] / input_size[1]
        ty = max_shift[0] / input_size[0]
        self.affine = K.RandomAffine(
            degrees=15.0, translate=(tx, ty), scale=(0.9, 1.1),
            p=1.0, same_on_batch=False, padding_mode="zeros")
        self.gamma = K.RandomGamma(gamma=(0.8, 1.25), p=0.5, same_on_batch=False)

    def forward(self, x):
        mx = x.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        x = x / mx
        x = self.affine(x)
        x = self.gamma(x.clamp(0.0, 1.0))
        return x


def gpu_standardize(x):
    """z-score par image sur GPU (entree uint8 ou float, cf sweep gpu_norm)."""
    x = x.float()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp(min=1e-5)
    return (x - m) / s
