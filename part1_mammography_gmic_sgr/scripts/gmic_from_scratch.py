"""
GMIC reconstruit « à la main » — sans la classe `GMIC` ni le pattern
`AbstractMILUnit / parent_module` de NYU.

Objectif pédagogique : montrer que le pipeline GMIC tient dans une poignée
de `nn.Module` standards et de fonctions pures. Seules les briques neuronales
génériques (ResNetV1, ResNetV2, BasicBlockV1, BasicBlockV2) sont réutilisées
depuis `src.modeling.modules` parce qu'elles sont déjà disséquées dans les
notebooks `basicblock_gmic.qmd` et `globalnetwork_gmic.qmd`.

Pipeline (une image 2944x1920 en entrée, scalaires par classe en sortie) :

    x  ──► GlobalBranch (ResNetV2 16 filtres + Conv1x1)  ──► h_g, saliency
                                                            │
                                                            ├─► top_t_percent ─► y_global
                                                            │
                                                            └─► retrieve_roi_greedy
                                                                  │
                                                                  ▼
                                               K crops 256x256 (extract_crops)
                                                                  │
                                                                  ▼
                                      LocalBranch (ResNetV1 64 filtres + GAP) ─► h_crops (K, 512)
                                                                  │
                                                                  ▼
                                       GatedAttention (Ilse 2018) ─► z, alpha, y_local
                                                                  │
                                                                  ▼
                                         Fusion (GMP + concat + Linear + sigmoid) ─► y_fusion

Les poids NYU (`sample_model_{1..5}.p`) sont chargeables via `load_nyu_weights()`
qui remappe les 258 clés utiles du checkpoint vers la structure de ce fichier.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rend le package `src` importable quand on exécute ce fichier en standalone
# depuis la racine du projet.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_GMIC_DIR = os.path.join(_PROJECT_ROOT, "GMIC")
if _GMIC_DIR not in sys.path:
    sys.path.insert(0, _GMIC_DIR)

from src.modeling.modules import (  # noqa: E402
    BasicBlockV1,
    BasicBlockV2,
    ResNetV1,
    ResNetV2,
)
from src.utilities import tools  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Global branch : backbone ResNetV2(16) + tête de saillance 1x1
# ═════════════════════════════════════════════════════════════════════════════

class SaliencyHead(nn.Module):
    """Conv 1x1 (sans biais) + sigmoid.

    Prend la carte de features `h_g` (N, 256, 46, 30) et produit une carte de
    saillance (N, num_classes, 46, 30) avec une probabilité par case et par
    classe. Équivalent de `PostProcessingStandard` dans le code NYU.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, h_g: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(h_g))


class GlobalBranch(nn.Module):
    """ResNetV2 léger (16 filtres initiaux) + SaliencyHead.

    Reproduit la configuration de `GlobalNetwork` dans le code NYU
    (voir `modules.py:291-301`). Les 5 blocs résiduels avec strides
    [1, 2, 2, 2, 2] et growth_factor=2 produisent la réduction spatiale
    2944x1920 → 46x30 et l'expansion canaux 1 → 256.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNetV2(
            input_channels=1,
            num_filters=16,
            first_layer_kernel_size=(7, 7),
            first_layer_conv_stride=2,
            first_layer_padding=3,
            first_pool_size=3,
            first_pool_stride=2,
            first_pool_padding=0,
            blocks_per_layer_list=[2, 2, 2, 2, 2],
            block_strides_list=[1, 2, 2, 2, 2],
            block_fn=BasicBlockV2,
            growth_factor=2,
        )
        self.head = SaliencyHead(in_channels=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_g = self.backbone(x)
        saliency = self.head(h_g)
        return h_g, saliency


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Agrégation Top-T% : carte 2D → scalaire par classe
# ═════════════════════════════════════════════════════════════════════════════

def top_t_percent(saliency: torch.Tensor, t: float) -> torch.Tensor:
    """Moyenne des `t*H*W` plus grandes activations de la carte de saillance.

    saliency : (N, C, H, W) — valeurs dans [0, 1] (sortie du sigmoid)
    t        : fraction (par ex. 0.02 pour top-2 %)
    return   : (N, C) — un score par classe
    """
    N, C, H, W = saliency.shape
    flat = saliency.view(N, C, -1)
    k = int(round(H * W * t))
    topk = flat.topk(k, dim=2)[0]
    return topk.mean(dim=2)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Sélection gloutonne des ROI + extraction des crops
# ═════════════════════════════════════════════════════════════════════════════

def retrieve_roi_greedy(
    x_original: torch.Tensor,
    saliency: torch.Tensor,
    K: int,
    crop_shape: Tuple[int, int],
    cam_size: Tuple[int, int],
    gpu_number: int | None,
) -> np.ndarray:
    """Sélection gloutonne de K fenêtres sur la carte de saillance.

    À chaque itération, cherche la fenêtre (adaptée à l'échelle `cam_size`)
    qui maximise la somme d'activation, enregistre son coin supérieur gauche,
    masque la région puis recommence. Reproduit la logique de
    `RetrieveROIModule.forward` (modules.py:350-388).

    Retourne un np.array de shape (N, K, 2) : coordonnées (row, col) dans
    la grille basse résolution `cam_size`.
    """
    _, _, H, W = x_original.size()
    h, w = cam_size

    # Fenêtre adaptée de `crop_shape` (taille originale) à `cam_size` (grille).
    cx = int(np.round(crop_shape[0] * h / H))
    cy = int(np.round(crop_shape[1] * w / W))
    win = (cx, cy)

    # Normalisation min-max par canal puis somme sur les canaux → carte 1D
    # utilisable par `get_max_window`.
    current = saliency
    N, C, _, _ = current.shape
    max_vals = current.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    min_vals = current.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
    normalized = (current - min_vals) / (max_vals - min_vals)
    current = normalized.sum(dim=1, keepdim=True)

    positions: List[torch.Tensor] = []
    for _ in range(K):
        pos = tools.get_max_window(current, win, "avg")
        positions.append(pos)
        mask = tools.generate_mask_uplft(current, win, pos, gpu_number)
        current = current * mask
    return torch.cat(positions, dim=1).data.cpu().numpy()


def convert_crop_position(
    locs_small: np.ndarray,
    cam_size: Tuple[int, int],
    x_original: torch.Tensor,
) -> np.ndarray:
    """Projette les coordonnées (row, col) de la grille `cam_size` vers la
    grille originale de `x_original`. Équivalent de
    `GMIC._convert_crop_position` (gmic.py:61-85).
    """
    h, w = cam_size
    _, _, H, W = x_original.size()

    prop_x = locs_small[:, :, 0] / h
    prop_y = locs_small[:, :, 1] / w
    assert np.max(prop_x) <= 1.0 and np.min(prop_x) >= 0.0
    assert np.max(prop_y) <= 1.0 and np.min(prop_y) >= 0.0

    inter_x = np.expand_dims(np.around(prop_x * H), -1)
    inter_y = np.expand_dims(np.around(prop_y * W), -1)
    return np.concatenate([inter_x, inter_y], axis=-1)


def extract_crops(
    x_original: torch.Tensor,
    locations: np.ndarray,
    crop_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Découpe physique des K crops à `crop_shape` depuis `x_original`.

    Équivalent de `GMIC._retrieve_crop` (gmic.py:87-108).
    Retourne un tenseur (N, K, crop_h, crop_w).
    """
    N, K, _ = locations.shape
    crop_h, crop_w = crop_shape
    out = torch.ones((N, K, crop_h, crop_w), device=device)
    for i in range(N):
        for j in range(K):
            tools.crop_pytorch(
                x_original[i, 0, :, :],
                crop_shape,
                locations[i, j, :],
                out[i, j, :, :],
                method="upper_left",
            )
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — Local branch : ResNetV1(64) + Global Average Pooling
# ═════════════════════════════════════════════════════════════════════════════

class LocalBranch(nn.Module):
    """ResNet-18 standard (64 filtres initiaux) pour un patch 256x256 isolé.

    Le patch est dupliqué sur 3 canaux (convention NYU), passé dans le ResNet
    puis écrasé spatialement par une moyenne globale → vecteur (N, 512).
    Équivalent de `LocalNetwork` (modules.py:391-413).
    """

    def __init__(self):
        super().__init__()
        self.resnet = ResNetV1(64, BasicBlockV1, [2, 2, 2, 2], 3)

    def forward(self, x_crop: torch.Tensor) -> torch.Tensor:
        # x_crop : (N*K, 1, 256, 256) → expand → (N*K, 3, 256, 256)
        feat_map = self.resnet(x_crop.expand(-1, 3, -1, -1))
        return feat_map.mean(dim=2).mean(dim=2)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 — Gated Attention (Ilse et al. 2018)
# ═════════════════════════════════════════════════════════════════════════════

class GatedAttention(nn.Module):
    """Mécanisme d'attention « gated » de Ilse et al. 2018.

    alpha_k ∝ exp( w^T ( tanh(V h_k) ⊙ sigmoid(U h_k) ) )
    z       = Σ alpha_k · h_k
    y_local = sigmoid( classifier · z )
    """

    def __init__(self, hidden_dim: int = 512, attn_dim: int = 128,
                 num_classes: int = 2):
        super().__init__()
        self.V = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.U = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h : (N, K, hidden_dim)
        N, K, D = h.shape
        h_flat = h.view(N * K, D)
        gate = torch.tanh(self.V(h_flat)) * torch.sigmoid(self.U(h_flat))
        score = self.w(gate).view(N, K)
        alpha = F.softmax(score, dim=1)
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)            # (N, D)
        y_local = torch.sigmoid(self.classifier(z))          # (N, num_classes)
        return z, alpha, y_local


# ═════════════════════════════════════════════════════════════════════════════
# Stage 6 — Fusion : Global Max Pool sur h_g + concat + Linear + sigmoid
# ═════════════════════════════════════════════════════════════════════════════

class Fusion(nn.Module):
    """Réconcilie la vue globale et la vue locale en un score final.

    (a) Global Max Pooling spatial sur `h_g` (N, 256, H, W) → (N, 256)
    (b) concat avec z (N, 512) → (N, 768)
    (c) Linear(768, num_classes) + sigmoid indépendant par classe.
    """

    def __init__(self, global_dim: int = 256, local_dim: int = 512,
                 num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(global_dim + local_dim, num_classes)

    def forward(self, h_g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g, _ = h_g.max(dim=2)
        g, _ = g.max(dim=2)                  # (N, 256)
        concat = torch.cat([g, z], dim=1)    # (N, 768)
        return torch.sigmoid(self.linear(concat))


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrateur — remplace la classe GMIC
# ═════════════════════════════════════════════════════════════════════════════

class ScratchGMIC(nn.Module):
    """Pipeline GMIC reconstruit avec des `nn.Module` standards.

    Contrairement à la classe `GMIC` de NYU, aucun module n'attache ses
    paramètres sur un parent : chaque étage est autonome. Les intermédiaires
    sont stockés comme attributs pour inspection directe sans monkeypatch.

    Attributs disponibles après `forward(x)` :
      h_g, saliency, y_global, locs_small, locations, crops,
      h_crops, z, alpha, y_local, y_fusion.
    """

    def __init__(
        self,
        K: int = 6,
        crop_shape: Tuple[int, int] = (256, 256),
        cam_size: Tuple[int, int] = (46, 30),
        percent_t: float = 0.02,
        num_classes: int = 2,
        device_type: str = "gpu",
        gpu_number: int = 0,
    ):
        super().__init__()
        self.K = K
        self.crop_shape = crop_shape
        self.cam_size = cam_size
        self.percent_t = percent_t
        self.device_type = device_type
        self.gpu_number = gpu_number if device_type == "gpu" else None

        self.global_net = GlobalBranch(num_classes=num_classes)
        self.local_net = LocalBranch()
        self.attention = GatedAttention(
            hidden_dim=512, attn_dim=128, num_classes=num_classes
        )
        self.fusion = Fusion(
            global_dim=256, local_dim=512, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 — Global branch
        self.h_g, self.saliency = self.global_net(x)

        # Stage 2 — Top-T% aggregation
        self.y_global = top_t_percent(self.saliency, self.percent_t)

        # Stage 3 — ROI selection + physical crop extraction
        self.locs_small = retrieve_roi_greedy(
            x, self.saliency, self.K, self.crop_shape,
            self.cam_size, self.gpu_number,
        )
        self.locations = convert_crop_position(
            self.locs_small, self.cam_size, x
        )
        self.crops = extract_crops(x, self.locations, self.crop_shape, x.device)

        # Stage 4 — Local branch on each crop
        N, K, Hc, Wc = self.crops.shape
        crops_flat = self.crops.view(N * K, Hc, Wc).unsqueeze(1)
        self.h_crops = self.local_net(crops_flat).view(N, K, -1)

        # Stage 5 — Gated attention
        self.z, self.alpha, self.y_local = self.attention(self.h_crops)

        # Stage 6 — Fusion
        self.y_fusion = self.fusion(self.h_g, self.z)
        return self.y_fusion


# ═════════════════════════════════════════════════════════════════════════════
# Chargement des poids NYU — remapping explicite des 258 clés utiles
# ═════════════════════════════════════════════════════════════════════════════

# Mapping (préfixe_NYU → préfixe_scratch). Appliqué au premier match, dans
# l'ordre de la liste (les préfixes les plus longs en premier pour éviter
# qu'une règle courte ne préempte une règle plus spécifique).
NYU_TO_SCRATCH: List[Tuple[str, str]] = [
    ("left_postprocess_net.gn_conv_last.", "global_net.head.conv."),
    ("ds_net.",            "global_net.backbone."),
    ("dn_resnet.",         "local_net.resnet."),
    ("mil_attn_V.",        "attention.V."),
    ("mil_attn_U.",        "attention.U."),
    ("mil_attn_w.",        "attention.w."),
    ("classifier_linear.", "attention.classifier."),
    ("fusion_dnn.",        "fusion.linear."),
]

# Clés NYU qu'on ignore volontairement (poids présents dans le checkpoint
# mais non utilisés par le `forward` de GMIC — reliquat de code).
NYU_SKIP_PREFIXES: Tuple[str, ...] = ("shared_rep_filter.",)


def remap_nyu_state_dict(state_dict: Dict[str, torch.Tensor]
                         ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Transforme un state_dict NYU en un state_dict compatible `ScratchGMIC`.

    Retourne (remapped, skipped) : le dict remappé et la liste des clés ignorées.
    """
    remapped: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in NYU_SKIP_PREFIXES):
            skipped.append(k)
            continue
        new_k = k
        for src, dst in NYU_TO_SCRATCH:
            if k.startswith(src):
                new_k = dst + k[len(src):]
                break
        else:
            # Aucune règle n'a matché → on laisse tel quel, `load_state_dict`
            # signalera l'anomalie en tant que clé inattendue.
            pass
        remapped[new_k] = v
    return remapped, skipped


def load_nyu_weights(
    model: ScratchGMIC,
    state_dict_path: str,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """Charge un checkpoint NYU (.p) dans un `ScratchGMIC`.

    Retourne (missing, unexpected, skipped).
    """
    raw = torch.load(state_dict_path, map_location=device, weights_only=False)
    remapped, skipped = remap_nyu_state_dict(raw)
    result = model.load_state_dict(remapped, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if verbose:
        print(f"Chargé : {len(remapped)} clés remappées depuis "
              f"{os.path.basename(state_dict_path)}")
        print(f"  ignorées (non utilisées par GMIC)  : {len(skipped)}")
        print(f"  manquantes (attendues mais absentes) : {len(missing)}")
        print(f"  inattendues (présentes mais non utilisées) : {len(unexpected)}")
        if missing:
            print("   → missing :", missing[:5], "..." if len(missing) > 5 else "")
        if unexpected:
            print("   → unexpected :", unexpected[:5],
                  "..." if len(unexpected) > 5 else "")
    return missing, unexpected, skipped


# ═════════════════════════════════════════════════════════════════════════════
# Auto-test en standalone
# ═════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    import imageio.v2 as imageio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = ScratchGMIC(
        K=6,
        crop_shape=(256, 256),
        cam_size=(46, 30),
        percent_t=0.02,
        num_classes=2,
        device_type="gpu" if device.type == "cuda" else "cpu",
        gpu_number=0,
    ).to(device).eval()

    ckpt = os.path.join(_PROJECT_ROOT, "GMIC", "models", "sample_model_1.p")
    load_nyu_weights(model, ckpt, device=device)

    img_path = os.path.join(
        _PROJECT_ROOT,
        "data/preprocess_image/sample_docker/cropped_images/10226/530620473.png",
    )
    img = imageio.imread(img_path).astype(np.float32)
    img = (img - img.mean()) / max(img.std(), 1e-5)
    x = torch.tensor(img[None, None]).to(device)

    with torch.no_grad():
        y_fusion = model(x)

    print("\nShapes des intermédiaires :")
    print(f"  h_g        : {tuple(model.h_g.shape)}")
    print(f"  saliency   : {tuple(model.saliency.shape)}")
    print(f"  y_global   : {tuple(model.y_global.shape)}  "
          f"→ malin={model.y_global[0,1].item():.4f}")
    print(f"  locations  : {model.locations.shape}")
    print(f"  crops      : {tuple(model.crops.shape)}")
    print(f"  h_crops    : {tuple(model.h_crops.shape)}")
    print(f"  alpha      : {tuple(model.alpha.shape)}  "
          f"(somme={model.alpha.sum().item():.3f})")
    print(f"  z          : {tuple(model.z.shape)}")
    print(f"  y_local    : {tuple(model.y_local.shape)}  "
          f"→ malin={model.y_local[0,1].item():.4f}")
    print(f"  y_fusion   : {tuple(y_fusion.shape)}  "
          f"→ malin={y_fusion[0,1].item():.4f}")


if __name__ == "__main__":
    _demo()
