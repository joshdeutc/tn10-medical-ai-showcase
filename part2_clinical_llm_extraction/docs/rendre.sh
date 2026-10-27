#!/bin/bash
# Renders the six documents to HTML and PDF in docs/_sortie/, one file per chapter.
#
# The interpreter is forced: rendering executes the code blocks, and .venv-docs is the
# only environment that carries neither torch nor unsloth — hence the only one that can
# be modified safely while a GPU run is using .venv312.
#
#   ./rendre.sh              HTML + PDF
#   ./rendre.sh --to pdf     PDF only
#   ./rendre.sh --to html    HTML only
set -euo pipefail
cd "$(dirname "$0")"

RACINE="$(cd .. && pwd)"
export QUARTO_PYTHON="$RACINE/.venv-docs/bin/python"
export PATH="$HOME/.local/bin:$PATH"

[ -x "$QUARTO_PYTHON" ] || {
  echo "rendering environment missing. Create it with:"
  echo "  uv venv --python 3.12 .venv-docs"
  echo "  uv pip install --python .venv-docs/bin/python jupyter pandas openpyxl"
  exit 1
}
command -v quarto >/dev/null || { echo "quarto not found in PATH"; exit 1; }

quarto render "$@"
echo
echo "output in docs/_sortie/ :"
ls -1sh _sortie/*.html _sortie/*.pdf 2>/dev/null || true
