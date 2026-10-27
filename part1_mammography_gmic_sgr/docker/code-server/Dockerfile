FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    PATH=/root/.local/bin:/workspace/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git openssh-client sudo \
        build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx

RUN mkdir -p /root/.config/code-server \
 && printf 'bind-addr: 0.0.0.0:8080\nauth: password\ncert: false\n' \
        > /root/.config/code-server/config.yaml

RUN code-server \
    --install-extension ms-python.python \
    --install-extension ms-toolsai.jupyter \
    --install-extension charliermarsh.ruff \
 || true

WORKDIR /workspace

EXPOSE 8080

CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "/workspace"]
