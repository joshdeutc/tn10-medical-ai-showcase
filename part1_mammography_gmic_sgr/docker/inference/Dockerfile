# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Image AUTONOME GMIC — TOUT-EN-UN : preprocess, inférence, training, éval,
# abstention, notebooks. Tout le code + les poids NYU + un sample brut sont
# embarqués. Les dépendances sont figées depuis uv.lock (parité EXACTE avec
# l'environnement de dev), torch/torchvision installés dans la variante choisie.
#
# Build CPU (par défaut, portable, sans libs CUDA) :
#   docker build -f docker/inference/Dockerfile -t gmic-inference:cpu .
#
# Build GPU (NVIDIA CUDA 12.1 — nécessite nvidia-container-toolkit) :
#   docker build -f docker/inference/Dockerfile \
#       --build-arg TORCH_VARIANT=cu121 -t gmic-inference:gpu .
#
# Utilisation (voir `entrypoint.sh` ou `docker run --rm gmic-inference:cpu help`) :
#   docker run --rm gmic-inference:cpu                       # démo preprocess+inférence
#   docker run --rm gmic-inference:cpu preprocess --help
#   docker run --rm gmic-inference:cpu infer --output-dir ...
#   docker run --rm gmic-inference:cpu eval --predictions ...
#   docker run --rm gmic-inference:cpu train --epochs 5
#   docker run --rm gmic-inference:cpu smoke                 # auto-test de l'image
#   docker run --rm gmic-inference:cpu bash                  # shell interactif
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Variante torch : "cpu" (défaut) ou "cu121" (GPU NVIDIA).
ARG TORCH_VARIANT=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# libGL + libglib : requis par opencv-python-headless. curl : pour installer uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# 1) torch / torchvision depuis l'index choisi (CPU ou CUDA 12.1). Les wheels
#    cu121 embarquent les libs nvidia-* ; les wheels cpu n'en ont pas besoin.
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
        torch==2.4.1 torchvision==0.19.1

# 2) Toutes les autres dépendances, FIGÉES depuis uv.lock (mêmes versions que
#    l'env de dev). On part de pyproject.toml + uv.lock seuls (layer cachable).
#      --no-emit-project : on n'installe pas le paquet local (code copié + PYTHONPATH)
#      --no-emit-package torch/torchvision : déjà installés à l'étape 1
#      grep -v nvidia/triton : deps CUDA transitives du lock, inutiles ici
#        (variante cu121 : récupérées par le wheel torch de l'étape 1).
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-hashes --no-emit-project \
        --no-emit-package torch --no-emit-package torchvision \
        --format requirements.txt \
      | grep -vE '^(nvidia-[a-z0-9-]+|triton)==' \
      > /tmp/requirements.txt \
 && uv pip install --system --no-cache -r /tmp/requirements.txt

# 3) Code + poids GMIC + images BRUTES (sample) — COPY sélectif.
COPY scripts/ ./scripts/
COPY utils/ ./utils/
COPY fine_tuning/ ./fine_tuning/
COPY abstention_module/ ./abstention_module/
COPY GMIC/src/ ./GMIC/src/
COPY GMIC/models/ ./GMIC/models/
COPY GMIC/LICENSE GMIC/NOTICE ./GMIC/
COPY data/raw/sample/train_images/ ./data/raw/sample/train_images/
COPY data/raw/sample/train.csv ./data/raw/sample/train.csv

COPY docker/inference/entrypoint.sh docker/inference/smoke_test.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/smoke_test.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["demo"]
