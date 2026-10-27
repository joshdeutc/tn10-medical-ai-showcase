# 🐳 Utiliser l'image Docker GMIC

Cette image est **autonome** : le code, les poids NYU et un échantillon de
8 mammographies brutes sont **embarqués dedans**. On peut donc lancer tout le
pipeline (preprocessing → inférence → évaluation → abstention) **sans cloner le
repo ni fournir de données**.

> ℹ️ Rappel express : une **image** est un disque figé prêt à l'emploi ; un
> **conteneur** est une exécution vivante de cette image. L'image n'est pas un
> fichier de ton projet, elle est stockée par le *daemon* Docker (`docker images`
> pour la lister).

---

## 1. Récupérer l'image

### Option A — la construire soi-même (si tu as le repo + les poids)

```bash
make docker-build        # image CPU (portable)  → gmic-inference:cpu
make docker-build-gpu    # image GPU (CUDA 12.1)  → gmic-inference:gpu
```

> ⚠️ Le build a besoin de `GMIC/models/*.p` (les poids) et de `data/raw/sample/`,
> qui sont **gitignorés** : ils ne sont pas dans le repo Git. Si tu n'as que le
> repo cloné sans ces fichiers, utilise l'option B.

### Option B — recevoir une image déjà construite

```bash
# Via un registre (Docker Hub / GitHub Container Registry)
docker pull ghcr.io/<compte>/gmic-inference:cpu

# Via un fichier (sans registre) : l'autre personne t'envoie un .tar.gz
docker load < gmic-inference-cpu.tar.gz
```

---

## 2. Lancer — le menu de commandes

L'image expose un **menu de sous-commandes**. Forme générale :

```bash
docker run [options] gmic-inference:cpu <commande> [arguments]
```

| Commande | Effet |
|---|---|
| *(rien)* ou `demo` | preprocess + inférence sur le sample embarqué |
| `pipeline` | démo **+ évaluation** (preprocess → inférence → éval) |
| `preprocess [args]` | `utils/preprocess.py` (crop + resize) |
| `infer [args]` | `scripts/inference.py` (inférence GMIC) |
| `eval [args]` | `scripts/eval_gmic.py` (métriques + abstention) |
| `train [args]` | `fine_tuning/train_resnet.py` (entraînement) |
| `test` | tests unitaires (pytest) |
| `smoke` | auto-test complet de l'image |
| `help` | affiche l'aide du menu |
| `bash` / `python ...` | exécuté tel quel (pass-through) |

```bash
docker run --rm gmic-inference:cpu help          # voir toutes les commandes
docker run --rm gmic-inference:cpu demo          # la démo (commande par défaut)
docker run --rm gmic-inference:cpu eval --help   # l'aide d'un script précis
docker run --rm -it gmic-inference:cpu bash      # shell interactif dans l'image
```

Lancer **n'importe quel** script (forme brute) :

```bash
docker run --rm gmic-inference:cpu python utils/preprocess.py --input-dir data/raw/sample --output-dir /tmp/out
```

> Avec le pass-through (`python scripts/X.py`), **aucune valeur par défaut**
> n'est injectée : tu dois fournir tous les arguments requis, comme en local.
> Les raccourcis du menu (`demo`, `infer`…), eux, remplissent les défauts utiles.

---

## 3. ⚠️ Où atterrissent les résultats (le point clé)

Un conteneur est **isolé** de ta machine. Par défaut, tout ce que les scripts
écrivent reste **dans le conteneur** — et avec `--rm`, c'est **détruit** à la fin.

```
   TA MACHINE                         LE CONTENEUR (éphémère avec --rm)
   /home/toi/projet/                  /app/
   (rien de nouveau)  ←──✗            ├── data/preprocess_image/...  ← créé ICI
                                      └── inference/runs/.../         ← puis DÉTRUIT
```

Pour que les sorties atterrissent **sur ton disque**, on **monte un dossier**
avec `-v chemin_hôte:chemin_conteneur` :

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/inference:/app/inference" \
  gmic-inference:cpu demo
```

Ici `/app/data` et `/app/inference` du conteneur **sont** tes dossiers `data/` et
`inference/` : les fichiers produits restent chez toi après la fin du conteneur.

| Commande | Résultats |
|---|---|
| `docker run --rm gmic-inference:cpu demo` | dans le conteneur → ❌ perdus |
| `docker run --rm -v "$(pwd)/inference:/app/inference" gmic-inference:cpu demo` | dans `inference/` chez toi → ✅ conservés |

> 🔑 L'image contient le **code et l'environnement** (figés) ; tout ce qui doit
> **entrer ou sortir** (tes données, tes résultats) passe par un `-v`.

---

## 4. Utiliser ses propres données

Monte ton dossier d'images brutes (entrée) **et** un dossier de sortie :

```bash
docker run --rm \
  -v /chemin/mes_mammos:/app/data/raw/perso \
  -v "$(pwd)/inference:/app/inference" \
  gmic-inference:cpu \
  python utils/preprocess.py --input-dir data/raw/perso --output-dir data/preprocess_image/perso

# puis l'inférence sur le dataset préprocessé
docker run --rm \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/inference:/app/inference" \
  gmic-inference:cpu \
  python scripts/inference.py --output-dir data/preprocess_image/perso
```

Les prédictions sont écrites dans un **dossier de run structuré** (voir §6).

---

## 5. GPU NVIDIA

Deux choses sont nécessaires : **l'image GPU** ET le flag `--gpus all`.

```bash
make docker-build-gpu                                  # image cu121
docker run --rm --gpus all gmic-inference:gpu demo     # = make docker-run-gpu
```

| Commande | GPU utilisé ? |
|---|---|
| `docker run gmic-inference:cpu` | non (image CPU) |
| `docker run gmic-inference:gpu` *(sans `--gpus`)* | non (carte non branchée) |
| `docker run --gpus all gmic-inference:gpu` | **oui** |

> Prérequis hôte : [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
> `inference.py` détecte CUDA automatiquement (`torch.cuda.is_available()`).

---

## 6. Format des sorties d'inférence

Les prédictions vont dans un dossier de run **structuré** (même logique que
`fine_tuning/checkpoints/runs/`) :

```
inference/runs/<dataset>/<model_tag>/<timestamp>/
├── predictions.csv   # image_index, malignant_pred, benign_pred, malignant_label
├── meta.json         # traçabilité : modèle, dataset, source, device, date, git
└── README.md         # résumé humain + commande d'évaluation
```

Utiliser `--run-dir <chemin>` pour écrire dans un dossier précis (et **reprendre**
un run interrompu : les images déjà présentes dans `predictions.csv` sont skippées).

---

## 7. Partager son image

```bash
# Vers un registre
docker tag gmic-inference:cpu ghcr.io/<compte>/gmic-inference:cpu
docker push ghcr.io/<compte>/gmic-inference:cpu

# Vers un fichier transportable (~1,5 Go)
docker save gmic-inference:cpu | gzip > gmic-inference-cpu.tar.gz
#   l'autre personne :  docker load < gmic-inference-cpu.tar.gz
```

> 🔒 Rien ne remonte vers toi : chaque personne fait tourner l'image sur **sa**
> machine, ses résultats restent chez elle. Tu as juste fourni un « moule ».

---

## 8. Tester que l'image fonctionne

```bash
make docker-test     # build CPU + smoke-test complet de bout en bout
```

Le smoke-test enchaîne pytest → preprocess → inférence → évaluation sur le sample
et échoue (code ≠ 0) au moindre problème. C'est l'équivalent local de la CI.

---

## 9. Dépannage rapide

| Symptôme | Cause / solution |
|---|---|
| `--input-dir est requis` | tu as lancé `python utils/preprocess.py` sans argument → fournis `--input-dir`, ou utilise `demo` |
| « je ne retrouve pas mes résultats » | il manque un `-v` → ils sont restés dans le conteneur (détruit par `--rm`) |
| `could not select device driver ... gpu` | `--gpus all` sans nvidia-container-toolkit, ou image CPU |
| image absente de `docker images` | pas encore buildée (`make docker-build`) ou pas chargée (`docker load`) |
| build échoue sur `COPY GMIC/models` | poids absents (gitignorés) → récupère l'image construite (option B) |

Détail du build (build-args, ce qui est embarqué) :
[`docker/inference/Dockerfile`](../docker/inference/Dockerfile).
