#!/bin/sh
# Download facebook/sam3 → checkpoints/sam3.pt after Hugging Face access is approved.
set -e
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "ERROR: python3 not found. Install Python 3.12+ or create .venv with ./setup.sh" >&2
    exit 1
fi

if ! "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub into this Python ($PYTHON)..."
    "$PYTHON" -m pip install -U "huggingface_hub[cli]"
fi

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Not logged in via HF_TOKEN. If download fails with 401/403, run:"
    echo "  $PYTHON -m pip install -U 'huggingface_hub[cli]'"
    echo "  hf auth login"
    echo "  # (old name: huggingface-cli login)"
fi

echo "Downloading facebook/sam3 (sam3.pt). This is several GB."
PYTHONPATH=src "$PYTHON" - <<'EOF'
from models.common import download_checkpoint

path = download_checkpoint("checkpoints")
print(f"Ready: {path}")
EOF
