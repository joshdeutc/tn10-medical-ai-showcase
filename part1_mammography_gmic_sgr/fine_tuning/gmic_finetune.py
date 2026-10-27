"""
Wrapper de finetuning pour GMIC.

- GMICTrain : sous-classe de GMIC dont le forward retourne les 3 tetes
  (y_fusion, y_global, y_local) + la saliency map, et evite les copies CPU
  (.cpu().numpy() sur les patches) inutiles a l'entrainement.
- build_gmic     : construit le modele et charge un checkpoint pre-entraine
  (strict=False).
- default_parameters / param_groups : hyperparams archi + LR discriminatif.

PYTHONPATH : ce module suppose que le dossier GMIC/ du repo est dans sys.path
(le script d'entrainement s'en charge) afin que `from src.modeling...` resolve.
"""
import torch
import torch.nn as nn

from src.modeling.gmic import GMIC


class GMICTrain(GMIC):
    """GMIC avec un forward d'entrainement (multi-tete, sans buffers de visu)."""

    def forward(self, x_original):
        # --- branche globale ---
        h_g, self.saliency_map = self.global_network.forward(x_original)
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # --- region proposal (non differentiable, numpy) ---
        small_x_locations = self.retrieve_roi_crops.forward(
            x_original, self.cam_size, self.saliency_map)
        self.patch_locations = self._convert_crop_position(
            small_x_locations, self.cam_size, x_original)

        # --- extraction des patchs (pas de copie self.patches) ---
        crops_variable = self._retrieve_crop(
            x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        batch_size, num_crops, I, J = crops_variable.size()

        # --- augmentation papier : rotation aleatoire {0,90,180,270} de chaque patch ---
        #   Entrainement uniquement ; rot90 sur patch carre = sans perte d'info.
        #   Force le reseau local a ne pas memoriser une orientation fixe -> anti-overfit.
        if self.training and I == J:
            ks = torch.randint(0, 4, (batch_size, num_crops))
            crops_variable = torch.stack([
                torch.rot90(crops_variable[i, j], int(ks[i, j]), dims=(0, 1))
                for i in range(batch_size) for j in range(num_crops)
            ]).view(batch_size, num_crops, I, J)

        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)

        # --- branche locale + MIL attention ---
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # --- fusion ---
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        # --- dropout regularisation (train-only) sur l'entree du classifieur fusion ---
        p_drop = self.experiment_parameters.get("dropout", 0.0)
        if p_drop > 0:
            concat_vec = nn.functional.dropout(concat_vec, p=p_drop, training=self.training)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        return self.y_fusion, self.y_global, self.y_local, self.saliency_map


def default_parameters(percent_t=0.02, gpu_number=0, device_type="gpu", dropout=0.0):
    """Hyperparametres archi identiques a run_model.py (modele ResNet-18 NYU)."""
    return {
        "device_type": device_type,
        "gpu_number": gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "percent_t": percent_t,
        "dropout": dropout,
    }


def build_gmic(weights_path, parameters, device):
    model = GMICTrain(parameters)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[build_gmic] charge {weights_path}")
        if missing:
            print(f"  missing keys    ({len(missing)}): {missing}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected}")
    return model.to(device)


# Noms des sous-modules backbone (greffes sur le module parent par add_layers)
_BACKBONE_PREFIXES = ("ds_net", "dn_resnet")


def param_groups(model, lr_backbone, lr_head):
    """LR discriminatif : backbones (global ResNet + ResNet des patchs) bas,
    tetes (postprocess, attention, classifier, fusion) plus haut."""
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(_BACKBONE_PREFIXES):
            backbone.append(p)
        else:
            head.append(p)
    n_bb = sum(p.numel() for p in backbone)
    n_hd = sum(p.numel() for p in head)
    print(f"[param_groups] backbone={n_bb/1e6:.2f}M @ lr={lr_backbone} | "
          f"head={n_hd/1e6:.2f}M @ lr={lr_head}")
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]


def freeze_backbone(model):
    """Linear-probe : gele les deux backbones (ds_net global + dn_resnet local).
    requires_grad=False -> exclus de l'optimiseur (param_groups filtre) et plus
    de backward a travers eux. A combiner avec set_backbone_eval (BN figee)."""
    n = 0
    for name, p in model.named_parameters():
        if name.startswith(_BACKBONE_PREFIXES):
            p.requires_grad = False
            n += p.numel()
    print(f"[freeze_backbone] {n/1e6:.2f}M params backbone geles (requires_grad=False)")


def set_backbone_eval(model):
    """Met les modules backbone en .eval() (BN -> running stats figees, pas de
    derive sur le domaine cible). A rappeler apres chaque model.train()."""
    for name, module in model.named_modules():
        if name.startswith(_BACKBONE_PREFIXES):
            module.eval()


# --- Variante : geler SEULEMENT le backbone GLOBAL (ds_net) ---
# Diagnostic par tete (12 juin 2026) : la tete global voit son AUC val culminer
# des l'epoch 0 puis se degrader (le finetuning DISTORD ses features NYU, cf
# LP-FT Kumar 2022), alors que le backbone local (dn_resnet) encaisse mieux le
# finetuning. Regime intermediaire : ds_net gele, dn_resnet + tetes entraines.
_GLOBAL_PREFIXES = ("ds_net",)


def freeze_global(model):
    """Gele UNIQUEMENT le backbone global (ds_net). dn_resnet + tetes restent
    entrainables. A combiner avec set_global_eval (BN figee)."""
    n = 0
    for name, p in model.named_parameters():
        if name.startswith(_GLOBAL_PREFIXES):
            p.requires_grad = False
            n += p.numel()
    print(f"[freeze_global] {n/1e6:.2f}M params backbone GLOBAL (ds_net) geles ; "
          f"dn_resnet + tetes restent entrainables")


def set_global_eval(model):
    """Met le backbone GLOBAL (ds_net) en .eval() (BN figee). A rappeler apres
    chaque model.train()."""
    for name, module in model.named_modules():
        if name.startswith(_GLOBAL_PREFIXES):
            module.eval()


# --- Dropout DANS les backbones (global ds_net + local dn_resnet) ---
# GMIC n a pas de dropout natif dans ses ResNet. Pour regulariser les backbones
# SANS modifier le code GMIC (il reste un pointeur ~/GMIC en lecture seule), on
# pose un forward-hook sur ds_net et dn_resnet qui applique un dropout2d
# (spatial : zeroie des canaux entiers de la feature map) sur leur SORTIE.
# Train-only : F.dropout2d(..., training=module.training) -> off en .eval()
# (donc desactive aussi quand le backbone est gele via set_*_eval). Aucun
# parametre ajoute -> param_groups / checkpoints inchanges.
def add_backbone_dropout(model, p):
    """Injecte un dropout2d (spatial, train-only) sur la sortie des backbones
    global (ds_net) et local (dn_resnet). p<=0 -> no-op. Retourne les handles."""
    if not p or p <= 0:
        return []

    def _hook(module, inputs, output):
        return nn.functional.dropout2d(output, p=p, training=module.training)

    handles = []
    for name in _BACKBONE_PREFIXES:
        sub = getattr(model, name, None)
        if sub is None:
            print(f"[backbone_dropout] WARN: sous-module {name} introuvable")
            continue
        handles.append(sub.register_forward_hook(_hook))
        print(f"[backbone_dropout] dropout2d p={p} pose sur {name}")
    return handles
