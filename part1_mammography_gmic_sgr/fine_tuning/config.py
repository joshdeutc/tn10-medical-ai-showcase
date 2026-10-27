"""
Configuration centrale du fine-tuning.
Modifier ce fichier pour changer les chemins et hyperparamètres.
"""

import os

# ─── Chemins ────────────────────────────────────────────────────────────────

# Dossier racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossier du run — produit par preprocess.py + inference.py
RUN_DIR = os.path.join(PROJECT_ROOT, "data", "preprocess_image")

# data.pkl : liste des exams avec leurs labels et chemins d'images
EXAM_LIST_PATH = os.path.join(RUN_DIR, "data.pkl")

# cropped_512/ : images uint8 512×512 pré-resizées par utils/preresize_images.py
# (fallback sur cropped_images/ 1920×2944 si le cache 512 n'existe pas encore)
IMAGE_DIR = os.path.join(RUN_DIR, "cropped_512")

# Dossier des images haute résolution (1920×2944) pour les runs v2
IMAGE_DIR_LARGE = os.path.join(RUN_DIR, "cropped_images")

# Dossier où sauvegarder les checkpoints du modèle
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "fine_tuning", "checkpoints")

# ─── Split train / validation ────────────────────────────────────────────────

# Proportion des exams utilisée pour la validation (le reste = train)
VAL_SPLIT = 0.2

# Graine aléatoire pour reproductibilité du split
RANDOM_SEED = 42

# ─── Images ─────────────────────────────────────────────────────────────────

# Taille de redimensionnement (H, W) des images en entrée du modèle.
IMAGE_SIZE = (1472, 960)

# ─── Entraînement ────────────────────────────────────────────────────────────

# Taille de batch
BATCH_SIZE = 1

# Nombre d'epochs
NUM_EPOCHS = 50

# Taux d'apprentissage initial
LEARNING_RATE = 1e-4

# Poids L2 (régularisation)
WEIGHT_DECAY = 1e-5

# Nombre de workers pour le DataLoader
NUM_WORKERS = 8

# Device d'entraînement.
DEVICE = "cuda"
