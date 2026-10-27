"""
Dataset PyTorch pour les mammographies GMIC.

Unité = 1 exam = 4 vues (L-CC, L-MLO, R-CC, R-MLO)
Label  = cancer_label['malignant']  →  0 (sain) ou 1 (cancer)

DataLoaders exposés :
  - get_train_loader() : échantillonnage équilibré 50% cancer / 50% sain
  - get_val_loader()   : distribution réelle (non équilibrée), pour AUC-ROC fiable
"""

import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF

from fine_tuning.config import (
    EXAM_LIST_PATH,
    IMAGE_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    RANDOM_SEED,
)

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class MammographyExamDataset(Dataset):
    """
    Charge un exam de mammographie (4 vues PNG) et retourne un tenseur + label.

    Retourne :
        images : Tensor (4, H, W)  — une image par vue, normalisée
        label  : Tensor scalaire   — 0 ou 1
    """

    def __init__(self, exam_list: list, image_dir: str, image_size: tuple, augment: bool = False):
        """
        Args:
            exam_list  : liste de dicts issus de cropped_exam_list.pkl
            image_dir  : dossier racine des PNG ({image_dir}/{patient_id}/{image_id}.png)
            image_size : (H, W) cible pour le resize
            augment    : True pour le train (flip horizontal aléatoire)
        """
        self.exam_list = exam_list
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.exam_list)

    def __getitem__(self, idx):
        exam = self.exam_list[idx]
        label = int(exam["cancer_label"]["malignant"])

        images = []
        for view in VIEWS:
            img_tensor = self._load_view(exam, view)
            images.append(img_tensor)

        # Stack : (4, H, W)
        images = torch.stack(images, dim=0)

        # Augmentation uniquement sur le train : flip horizontal aléatoire
        if self.augment and torch.rand(1).item() > 0.5:
            images = TF.hflip(images)

        return images, torch.tensor(label, dtype=torch.float32)

    def _load_view(self, exam: dict, view: str) -> torch.Tensor:
        """
        Charge une vue PNG, redimensionne, normalise et retourne un Tensor (H, W).
        Si la vue est manquante, retourne un tenseur de zéros.
        """
        view_files = exam.get(view, [])

        if not view_files:
            H, W = self.image_size
            return torch.zeros(H, W, dtype=torch.float32)

        # Chemin : {image_dir}/{patient_id}/{image_id}.png
        rel_path = view_files[0]                          # ex: "10011/220375232"
        png_path = os.path.join(self.image_dir, rel_path + ".png")

        if not os.path.exists(png_path):
            H, W = self.image_size
            return torch.zeros(H, W, dtype=torch.float32)

        # Lecture et conversion float32
        import imageio
        image = np.array(imageio.imread(png_path), dtype=np.float32)

        # Les vues droites ont deja ete retournees lors du preprocessing.
        # Ici on applique uniquement resize + normalisation.

        # Redimensionnement via PIL (interp. bilinéaire)
        from PIL import Image
        pil_img = Image.fromarray(image)
        H, W = self.image_size
        pil_img = pil_img.resize((W, H), Image.BILINEAR)
        image = np.array(pil_img, dtype=np.float32)

        # Normalisation z-score par image (standard en mammographie)
        mean = image.mean()
        std = max(image.std(), 1e-5)
        image = (image - mean) / std

        return torch.tensor(image, dtype=torch.float32)


# ─── Split train / val ────────────────────────────────────────────────────────

def load_and_split(exam_list_path: str = EXAM_LIST_PATH):
    """
    Charge le pkl et sépare en train/val de manière stratifiée
    (respecte la proportion cancer/sain dans chaque split).

    Returns:
        train_exams, val_exams : listes de dicts
    """
    with open(exam_list_path, "rb") as f:
        all_exams = pickle.load(f)

    labels = [int(exam["cancer_label"]["malignant"]) for exam in all_exams]

    # Split stratifié si possible (nécessite ≥2 exemples par classe)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    can_stratify = n_pos >= 2 and n_neg >= 2

    if not can_stratify:
        print(f"[AVERTISSEMENT] Trop peu d'exemples pour un split stratifié "
              f"(cancer={n_pos}, sain={n_neg}). Split aléatoire utilisé.")

    train_exams, val_exams = train_test_split(
        all_exams,
        test_size=VAL_SPLIT,
        stratify=labels if can_stratify else None,
        random_state=RANDOM_SEED,
    )

    n_pos_train = sum(1 for e in train_exams if e["cancer_label"]["malignant"] == 1)
    n_neg_train = sum(1 for e in train_exams if e["cancer_label"]["malignant"] == 0)
    n_pos_val   = sum(1 for e in val_exams   if e["cancer_label"]["malignant"] == 1)
    n_neg_val   = sum(1 for e in val_exams   if e["cancer_label"]["malignant"] == 0)

    print(f"Train : {len(train_exams)} exams  |  cancer={n_pos_train}  sain={n_neg_train}")
    print(f"Val   : {len(val_exams)}  exams  |  cancer={n_pos_val}   sain={n_neg_val}")

    return train_exams, val_exams


# ─── WeightedRandomSampler (50/50) ───────────────────────────────────────────

def make_balanced_sampler(exam_list: list) -> WeightedRandomSampler:
    """
    Crée un WeightedRandomSampler qui tire autant de cancer que de sain
    à chaque epoch, quelle que soit la proportion réelle dans exam_list.
    """
    labels = [int(exam["cancer_label"]["malignant"]) for exam in exam_list]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    # Poids inversement proportionnels à la fréquence de la classe
    weight_pos = 1.0 / n_pos if n_pos > 0 else 0.0
    weight_neg = 1.0 / n_neg if n_neg > 0 else 0.0

    sample_weights = [
        weight_pos if label == 1 else weight_neg
        for label in labels
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ─── DataLoaders publics ─────────────────────────────────────────────────────

def get_train_loader(
    exam_list_path: str = EXAM_LIST_PATH,
    image_dir: str = IMAGE_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    DataLoader d'entraînement avec échantillonnage équilibré 50/50.
    Inclut l'augmentation (flip horizontal aléatoire).
    """
    train_exams, _ = load_and_split(exam_list_path)

    dataset = MammographyExamDataset(
        exam_list=train_exams,
        image_dir=image_dir,
        image_size=image_size,
        augment=True,
    )

    sampler = make_balanced_sampler(train_exams)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,           # remplace shuffle=True
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train loader : {len(dataset)} exams, batch_size={batch_size}, équilibré 50/50")
    return loader, train_exams


def get_val_loader(
    exam_list_path: str = EXAM_LIST_PATH,
    image_dir: str = IMAGE_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    DataLoader de validation — distribution RÉELLE (non équilibrée).
    Pas d'augmentation. Nécessaire pour un AUC-ROC non biaisé.
    """
    _, val_exams = load_and_split(exam_list_path)

    dataset = MammographyExamDataset(
        exam_list=val_exams,
        image_dir=image_dir,
        image_size=image_size,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,             # ordre fixe pour la reproductibilité
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Val loader   : {len(dataset)} exams, batch_size={batch_size}, distribution réelle")
    return loader, val_exams
