# Troubleshooting - Erreurs communes

## Problemes avec les images

### Image en couleur (RGB)

```
[FAIL] Image en COULEUR (RGB) : data/train_images/123/456.png
```

**Cause** : Le fichier n'est pas une mammographie. Les mammographies sont
toujours en niveaux de gris (1 seul canal). Une image RGB (3 canaux) est
probablement une photo classique (portrait, paysage, etc.).

**Solution** : Verifiez que vous utilisez les bonnes images. Les mammographies
viennent de machines medicales (format DICOM), pas d'Internet ou d'un appareil
photo.

---

### Image trop petite

```
[FAIL] Image trop petite : data/train_images/123/456.png (200x150)
```

**Cause** : Les mammographies font typiquement entre 2000x3000 et 5000x6000
pixels. Une image de quelques centaines de pixels est probablement :
- Une vignette (thumbnail)
- Une image web basse resolution
- Un screenshot

**Solution** : Utilisez les images DICOM originales du dataset RSNA, ou des
PNG convertis depuis DICOM en resolution complete.

---

### Image uniforme (toute noire / toute blanche)

```
[FAIL] Image presque uniforme : (min=0, max=2, ecart=2)
```

**Cause** : L'image est presque entierement noire ou blanche. Elle est
probablement corrompue ou le fichier est vide.

**Solution** : Re-telechargez l'image depuis Kaggle. Si le probleme persiste,
excluez ce patient du CSV.

---

### Image RGBA (avec transparence)

```
[FAIL] Image RGBA (avec transparence) : data/123/photo.png
```

**Cause** : L'image a 4 canaux (RGB + alpha). C'est typiquement une capture
d'ecran ou une image web, pas une mammographie.

**Solution** : Utilisez les images DICOM originales.

---

## Problemes avec le CSV

### Colonnes manquantes

```
[FAIL] Colonnes manquantes : {'laterality', 'view'}
```

**Cause** : Le CSV ne contient pas toutes les colonnes requises par GMIC.

**Colonnes obligatoires** :
- `patient_id` — identifiant unique du patient
- `image_id` — identifiant unique de l'image
- `laterality` — cote du sein : `L` (gauche) ou `R` (droite)
- `view` — type de vue : `CC` ou `MLO`
- `cancer` — label : `0` (sain) ou `1` (cancer)

**Solution** : Verifiez le format de votre CSV. Le CSV du dataset RSNA
(`train.csv`) contient toutes ces colonnes par defaut.

---

### Lateralites ou vues invalides

```
[FAIL] Lateralites invalides : {'Left'}
[FAIL] Vues invalides : {'cranio-caudal'}
```

**Cause** : Les valeurs ne correspondent pas au format attendu.

**Solution** : Les lateralites doivent etre `L` ou `R` (pas `Left`/`Right`).
Les vues doivent etre `CC` ou `MLO` (pas le nom complet).

---

### Vues manquantes pour certains patients

```
[WARN] 15/100 patients n'ont pas les 4 vues standard
```

**Cause** : GMIC s'attend a recevoir 4 vues par patient (L-CC, L-MLO, R-CC,
R-MLO). Certains patients n'ont qu'un seul cote, ou une seule vue par cote.

**Consequences possibles** :
- `IndexError: list index out of range` dans `run_model.py`
- Resultats biaises si on duplique les vues manquantes

**Solution** : Idealement, n'utilisez que les patients avec les 4 vues
completes. Sinon, le pipeline tentera de fonctionner mais les resultats
seront moins fiables.

---

## Problemes avec le pipeline

### AssertionError dans get_optimal_centers

```
assert np.all(br_y >= tl_y)
AssertionError
```

**Cause** : Les metadonnees du PKL (coordonnees `rightmost_points`,
`bottommost_points`) ne correspondent plus aux dimensions des images.
Cela arrive quand les images ont ete redimensionnees mais le PKL n'a
pas ete mis a jour.

**Solution** : Relancez le pipeline avec `--force-resize` :
```bash
make run ARGS="--force-resize"
```

---

### crop_mammogram ne produit rien

**Cause** : Le dossier `cropped_images/` existe deja. Le script GMIC
`crop_mammogram.py` quitte silencieusement si le dossier de sortie existe.

**Solution** : Le pipeline supprime automatiquement le dossier avant de
relancer le crop. Si le probleme persiste, forcez :
```bash
make run ARGS="--force-crop"
```

---

### Modeles GMIC manquants

```
[FAIL] Modeles GMIC manquants : ['sample_model_1.p', ...]
```

**Solution** : Telechargez les 5 fichiers `sample_model_1.p` a
`sample_model_5.p` depuis le depot GMIC officiel et placez-les dans
`GMIC/models/`.

---

## Lancer la validation

Avant de lancer le pipeline, verifiez vos donnees :

```bash
# Validation standard
python utils/validate_input.py --input-dir data/

# Validation stricte (warnings = erreurs)
python utils/validate_input.py --input-dir data/ --strict

# Avec un CSV specifique
python utils/validate_input.py --input-dir data/ --csv data/train.csv
```
