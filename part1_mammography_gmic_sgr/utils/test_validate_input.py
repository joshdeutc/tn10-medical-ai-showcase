"""
Tests unitaires pour la validation des donnees d'entree.
Lance avec : make test  ou  uv run pytest utils/test_validate_input.py -v
"""

import os
import csv
import tempfile
import shutil

import cv2
import numpy as np
import pytest

from utils.validate_input import (
    ValidationResult,
    check_csv,
    check_image,
)


@pytest.fixture
def tmp_dir():
    """Cree un dossier temporaire pour les tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def valid_csv(tmp_dir):
    """Cree un CSV valide minimal."""
    path = os.path.join(tmp_dir, "train.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
        w.writerow(["10001", "111", "L", "CC", 0])
        w.writerow(["10001", "112", "L", "MLO", 0])
        w.writerow(["10001", "113", "R", "CC", 0])
        w.writerow(["10001", "114", "R", "MLO", 0])
        w.writerow(["10002", "211", "L", "CC", 1])
        w.writerow(["10002", "212", "L", "MLO", 1])
        w.writerow(["10002", "213", "R", "CC", 1])
        w.writerow(["10002", "214", "R", "MLO", 1])
    return path


def make_grayscale_image(path, h=3000, w=2000, dtype=np.uint16):
    """Cree une image grayscale valide (simule une mammographie)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(100, 60000, (h, w), dtype=dtype)
    cv2.imwrite(path, img)
    return path


def make_rgb_image(path, h=3000, w=2000):
    """Cree une image couleur RGB (photo classique)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_tiny_image(path, h=400, w=350):
    """Cree une image trop petite (sous le seuil de 700px)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_white_image(path, h=3000, w=2000):
    """Cree une image toute blanche."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_large_image(path, h=6000, w=2000):
    """Cree une image avec une dimension > 5000px."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(10, 240, (h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_uint8_image(path, h=3000, w=2000):
    """Cree une image uint8 avec distribution uniforme entre 0 et 255."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_black_image(path, h=3000, w=2000):
    """Cree une image toute noire."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


# ── Tests CSV ────────────────────────────────────────────────────────────────

class TestCSV:
    def test_valid_csv(self, valid_csv):
        result = ValidationResult()
        df = check_csv(valid_csv, result)
        assert result.passed
        assert df is not None
        assert len(df) == 8

    def test_missing_csv(self):
        result = ValidationResult()
        df = check_csv("/nonexistent/train.csv", result)
        assert not result.passed
        assert df is None

    def test_csv_is_a_directory(self, tmp_dir):
        # Un dossier nomme train.csv (cas reel rencontre dans Downloads)
        fake_csv = os.path.join(tmp_dir, "train.csv")
        os.makedirs(fake_csv)
        result = ValidationResult()
        df = check_csv(fake_csv, result)
        assert not result.passed
        assert any("dossier" in e for e in result.errors)

    def test_missing_columns(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id"])
            w.writerow(["10001", "111"])
        result = ValidationResult()
        df = check_csv(path, result)
        assert not result.passed
        assert "Colonnes manquantes" in result.errors[0]

    def test_invalid_laterality(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_lat.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            w.writerow(["10001", "111", "Left", "CC", 0])
        result = ValidationResult()
        check_csv(path, result)
        assert not result.passed
        assert any("Lateralites invalides" in e for e in result.errors)

    def test_invalid_view(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_view.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            w.writerow(["10001", "111", "L", "XRAY", 0])
        result = ValidationResult()
        check_csv(path, result)
        assert not result.passed
        assert any("Vues invalides" in e for e in result.errors)


# ── Tests Images ─────────────────────────────────────────────────────────────

class TestImages:
    def test_valid_grayscale(self, tmp_dir):
        path = make_grayscale_image(os.path.join(tmp_dir, "img.png"))
        result = ValidationResult()
        assert check_image(path, result) is True
        assert result.passed

    def test_rgb_image_rejected(self, tmp_dir):
        path = make_rgb_image(os.path.join(tmp_dir, "color.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert not result.passed
        assert any("COULEUR" in e for e in result.errors)

    def test_tiny_image_warns(self, tmp_dir):
        path = make_tiny_image(os.path.join(tmp_dir, "tiny.png"))
        result = ValidationResult()
        assert check_image(path, result) is True
        assert any("petite" in w for w in result.warnings)

    def test_black_image_rejected(self, tmp_dir):
        path = make_black_image(os.path.join(tmp_dir, "black.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("uniforme" in e for e in result.errors)

    def test_corrupted_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "corrupt.png")
        with open(path, "w") as f:
            f.write("this is not an image")
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("illisible" in e for e in result.errors)

    def test_white_image_rejected(self, tmp_dir):
        path = make_white_image(os.path.join(tmp_dir, "white.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("uniforme" in e for e in result.errors)

    def test_large_image_warns(self, tmp_dir):
        path = make_large_image(os.path.join(tmp_dir, "large.png"))
        result = ValidationResult()
        ok = check_image(path, result)
        assert ok is True
        assert any("grande" in w for w in result.warnings)

    def test_uint8_valid(self, tmp_dir):
        """Une image uint8 avec pixels entre 0 et 255 doit etre acceptee."""
        path = make_uint8_image(os.path.join(tmp_dir, "uint8.png"))
        result = ValidationResult()
        ok = check_image(path, result)
        assert ok is True
        assert result.passed
        # Verifier que l'image lue est bien en entiers (dtype uint8)
        import cv2 as _cv2
        img = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
        assert img.dtype.kind == 'u'      # unsigned integer
        assert img.min() >= 0
        assert img.max() <= 255


