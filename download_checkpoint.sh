#!/bin/sh
# Download facebook/sam3 → checkpoints/sam3.pt after Hugging Face access is approved.
set -e
cd "$(dirname "$0")"

# huggingface_hub installs `hf` / `huggingface-cli` here; many shells omit it from PATH.
export PATH="$HOME/.local/bin:$PATH"

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
    "$PYTHON" -m pip install -U huggingface_hub
fi

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "No HF_TOKEN in the environment. If download returns 401, run:"
    echo "  ./login_hf.sh"
    echo "  # or: export HF_TOKEN=hf_..."
fi

echo "Downloading facebook/sam3 (sam3.pt). This is several GB."
"$PYTHON" - <<'EOF'
import os
import shutil

from huggingface_hub import hf_hub_download

repo, ckpt, cfg = "facebook/sam3", "sam3.pt", "config.json"
os.makedirs("checkpoints", exist_ok=True)
hf_hub_download(repo, cfg)
src = hf_hub_download(repo, ckpt)
dest = os.path.abspath(os.path.join("checkpoints", ckpt))
shutil.copy(src, dest)
print(f"Ready: {dest}")
EOF
