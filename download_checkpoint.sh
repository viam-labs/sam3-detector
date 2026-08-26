#!/bin/sh
# Download facebook/sam3 → checkpoints/sam3.pt after Hugging Face access is approved.
set -e
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Tip: if huggingface-cli login has not been run, export HF_TOKEN=hf_..."
fi

echo "Downloading facebook/sam3 (sam3.pt). This is several GB."
PYTHONPATH=src "$PYTHON" - <<'EOF'
from models.common import download_checkpoint

path = download_checkpoint("checkpoints")
print(f"Ready: {path}")
EOF
