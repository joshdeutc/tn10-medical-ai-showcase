"""
Dataset PyTorch pour le finetuning de GMIC.

GMIC traite UNE vue a la fois -> l'unite est l'IMAGE (la vue), pas l'exam.
  - entree : Tensor (1, 2944, 1920) float32, z-score par image
  - label  : Tensor (2,)  = [benign, malignant] du SEIN correspondant

Les PNG de cropped_images/ sont deja :
  - croppes/resizes a 2944x1920 (taille d'entree GMIC) -> pas de process_image/best_center
  - flippes (vues droites) -> orientation chest-wall-a-gauche pour toutes les vues
    => on NE refait PAS de flip horizontal (casserait la convention).

Split patient-level stratifie (pas de fuite de vues d'un meme patient entre train/val).
Sampler : WeightedRandomSampler equilibrant les vues avec lesion (benign OU
malignant) vs negatives, ~50/50 par epoch (recette papier GMIC, anti-collapse).
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

import imageio.v2 as imageio

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
INPUT_SIZE = (2944, 1920)  # (H, W)


def _side(view):
    return "left" if view.startswith("L") else "right"


def _exam_pid(exam):
    for v in VIEWS:
        fs = exam.get(v) or []
        if fs:
            return fs[0].split("/")[0]
    return None


def build_samples(exam_list, image_dir):
    """Aplati la liste d'exams en liste d'echantillons par vue."""
    samples = []
    for exam in exam_list:
        cl = exam["cancer_label"]
        pid = _exam_pid(exam)
        for v in VIEWS:
            fs = exam.get(v) or []
            if not fs:
                continue
            path = os.path.join(image_dir, fs[0] + ".png")
            if not os.path.exists(path):
                continue
            side = _side(v)
            ben = int(cl.get(f"{side}_benign", 0))
            mal = int(cl.get(f"{side}_malignant", 0))
            samples.append({
                "path": path,
                "view": v,
                "pid": pid,
                "label": np.array([ben, mal], dtype=np.float32),
                "malignant": mal,
            })
    return samples


class GMICViewDataset(Dataset):
    def __init__(self, samples, augment=False, input_size=INPUT_SIZE,
                 max_shift=(48, 32), to_uint8=False):
        self.samples = samples
        self.augment = augment
        self.input_size = input_size
        self.max_shift = max_shift  # (dy_max, dx_max) en pixels
        self.to_uint8 = to_uint8  # True -> dtype compact, normalisation reportee sur GPU
        # Detection de la profondeur du cache : PNG uint16 (cropped_16bit, valeurs
        # natives 12 bits <=4095) -> transfert int16 (torch n'a pas d'uint16) ;
        # PNG uint8 (cropped_images) -> transfert uint8. Le z-score GPU en aval
        # est insensible a l'echelle.
        self.bit16 = False
        if samples:
            im0 = np.asarray(imageio.imread(samples[0]["path"]))
            self.bit16 = im0.dtype == np.uint16

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        im = np.asarray(imageio.imread(path), dtype=np.float32)
        if im.shape != self.input_size:
            import cv2
            im = cv2.resize(im, (self.input_size[1], self.input_size[0]),
                            interpolation=cv2.INTER_CUBIC)
        return im

    def _augment(self, im):
        # Augmentation enrichie (anti-overfit, alignee papier GMIC). TRAIN only.
        # Geometrique : rotation +/-15, scale 0.9-1.1, translation +/-max_shift,
        #   le tout en UNE affine (cv2.warpAffine, fond noir = bg du crop mammo).
        #   PAS de flip horizontal : les vues sont deja normalisees chest-wall-a-
        #   gauche, un hflip produirait une orientation jamais vue au test.
        # Intensite : jitter GAMMA uniquement. Un brightness/contraste LINEAIRE
        #   serait annule par le z-score (GPU) -> seul un gamma non-lineaire survit.
        import cv2
        H, W = im.shape
        ang = np.random.uniform(-15.0, 15.0)
        scale = np.random.uniform(0.9, 1.1)
        mh, mw = self.max_shift
        dy = np.random.uniform(-mh, mh)
        dx = np.random.uniform(-mw, mw)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        im = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if np.random.rand() < 0.5:  # gamma relatif au max de l'image (scale-aware :
            # uint8 minmax-stretche -> max~255, identique a l'historique /255 ;
            # uint16 natif 12 bits -> max reel de l'image)
            gamma = np.random.uniform(0.8, 1.25)
            vr = float(im.max())
            if vr > 0:
                im = vr * np.power(np.clip(im, 0.0, vr) / vr, gamma)
        return im

    def __getitem__(self, idx):
        s = self.samples[idx]
        im = self._load(s["path"])
        if self.augment:
            im = self._augment(im)
        if self.to_uint8:
            # dtype compact (2-4x moins de RAM/bande passante que float32) ;
            # z-score fait sur GPU cote train
            if self.bit16:
                # valeurs natives 12 bits <=4095 : tiennent largement en int16
                img = torch.from_numpy(np.ascontiguousarray(
                    np.clip(im, 0.0, 32767.0).astype(np.int16))).unsqueeze(0)
            else:
                img = torch.from_numpy(np.ascontiguousarray(im.astype(np.uint8))).unsqueeze(0)
        else:
            # z-score par image (standard_normalize de GMIC)
            im = im - im.mean()
            im = im / max(im.std(), 1e-5)
            img = torch.from_numpy(np.ascontiguousarray(im)).unsqueeze(0)  # (1,H,W)
        label = torch.from_numpy(s["label"])
        return img, label


def load_and_split3(pkl_path, train_frac=2.0 / 3, abstain_frac=1.0 / 6,
                    val_frac=1.0 / 6, seed=42):
    """Split patient-level DISJOINT en 3 ensembles : train / abstention / val.

    L'abstention (selective prediction) doit s'entrainer sur des donnees
    disjointes du classifieur GMIC, sinon ses scores de confiance sont biaises.
    Stratifie sur "le patient a-t-il un malignant". Defaut : 1/3, 1/2, 1/6.
    """
    with open(pkl_path, "rb") as f:
        exams = pickle.load(f)

    # split patient-level stratifie par "patient a-t-il un malignant"
    pid_label = {}
    for e in exams:
        pid = _exam_pid(e)
        if pid is None:
            continue
        m = int(e["cancer_label"].get("malignant", 0))
        pid_label[pid] = max(pid_label.get(pid, 0), m)

    uniq = sorted(pid_label.keys())
    labels = [pid_label[p] for p in uniq]

    def _can_strat(lbls):
        npos = sum(lbls)
        return npos >= 2 and (len(lbls) - npos) >= 2

    # 1) extraire la validation (val_frac du total)
    rest_pids, val_pids = train_test_split(
        uniq, test_size=val_frac,
        stratify=labels if _can_strat(labels) else None,
        random_state=seed,
    )
    # 2) separer train vs abstention dans le reste.
    #    train doit valoir train_frac du TOTAL -> train_frac/(1-val_frac) du reste
    #    abstain_frac<=0 -> pas de held-out abstention, tout le reste part en
    #    train (la val extraite en 1) ne depend que de val_frac+seed -> reste
    #    IDENTIQUE aux runs avec abstention : comparables).
    if abstain_frac <= 0:
        train_pids, abstain_pids = list(rest_pids), []
    else:
        rest_labels = [pid_label[p] for p in rest_pids]
        train_size_rel = train_frac / (1.0 - val_frac)
        train_pids, abstain_pids = train_test_split(
            rest_pids, train_size=train_size_rel,
            stratify=rest_labels if _can_strat(rest_labels) else None,
            random_state=seed,
        )

    train_set, abstain_set, val_set = set(train_pids), set(abstain_pids), set(val_pids)
    train_ex = [e for e in exams if _exam_pid(e) in train_set]
    abstain_ex = [e for e in exams if _exam_pid(e) in abstain_set]
    val_ex = [e for e in exams if _exam_pid(e) in val_set]
    split_pids = {"train": sorted(train_set), "abstain": sorted(abstain_set),
                  "val": sorted(val_set)}
    return train_ex, abstain_ex, val_ex, split_pids


def _make_sampler(samples):
    # Recette papier GMIC : a chaque epoch, equilibrer "vue avec lesion
    # (benign OU malignant)" vs vue negative -> ~50/50 par epoch.
    # label = [benign, malignant] ; finding = au moins une des deux.
    finding = [int(s["label"][0] > 0 or s["label"][1] > 0) for s in samples]
    n_pos = sum(finding)
    n_neg = len(finding) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0.0
    w_neg = 1.0 / n_neg if n_neg else 0.0
    weights = [w_pos if f else w_neg for f in finding]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loaders(pkl_path, image_dir, batch_size=4, num_workers=8,
                 train_frac=2.0 / 3, abstain_frac=1.0 / 6, val_frac=1.0 / 6,
                 seed=42, augment=True, input_size=INPUT_SIZE, split_out=None,
                 to_uint8=True):
    train_ex, abstain_ex, val_ex, split_pids = load_and_split3(
        pkl_path, train_frac, abstain_frac, val_frac, seed)
    train_samples = build_samples(train_ex, image_dir)
    val_samples = build_samples(val_ex, image_dir)
    abstain_samples = build_samples(abstain_ex, image_dir)

    def _stats(samples, name):
        n = len(samples)
        npos = sum(s["malignant"] for s in samples)
        nben = sum(int(s["label"][0]) for s in samples)
        print(f"{name:9s}: {n} vues | malignant={npos} | benign={nben}")

    _stats(train_samples, "train")
    _stats(abstain_samples, "abstention")
    _stats(val_samples, "val")

    # Sauvegarde du split patient-level : le modele d'abstention DOIT reutiliser
    # exactement ces ensembles disjoints (sinon fuite train<->abstention).
    if split_out:
        import json
        payload = {"seed": seed,
                   "fracs": {"train": train_frac, "abstain": abstain_frac,
                             "val": val_frac},
                   "n_patients": {k: len(v) for k, v in split_pids.items()},
                   "pids": split_pids}
        with open(split_out, "w") as f:
            json.dump(payload, f)
        print(f"split patient-level (train/abstention/val) -> {split_out}")

    train_ds = GMICViewDataset(train_samples, augment=augment, input_size=input_size,
                               to_uint8=to_uint8)
    val_ds = GMICViewDataset(val_samples, augment=False, input_size=input_size,
                             to_uint8=to_uint8)

    # --- config optimale (sweep _gmic_sweep.py) : images en uint8 -> normalisation
    #     sur GPU cote train -> RAM bornee meme avec pin_memory + prefetch 2
    # En 16 bits (int16 = 2x les octets de l'uint8) l'empreinte RAM double ;
    # sur cette VM (94 Go SANS swap) ca part en reclaim/compaction (88% sys,
    # GPU affame, blocage de fait). Chemin FRUGAL en 16 bits : pas de
    # pin_memory (page-locked = non reclaimable) + prefetch 1. Le 8 bits garde
    # le chemin rapide du sweep (pin + prefetch 2). N'affecte PAS les resultats.
    frugal = train_ds.bit16
    loader_kw = dict(num_workers=num_workers, pin_memory=not frugal)
    if num_workers > 0:
        loader_kw.update(prefetch_factor=(1 if frugal else 2), persistent_workers=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=_make_sampler(train_samples),
        drop_last=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kw,
    )
    return train_loader, val_loader, train_samples, val_samples
