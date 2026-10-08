#!/bin/sh
# Download facebook/sam3 → checkpoints/sam3.pt after Hugging Face access is approved.
set -e
cd "$(dirname "$0")"
SAM3_ROOT="$(pwd)"
# shellcheck disable=SC1091
. ./load_hf_env.sh

# huggingface_hub installs `hf` here; many shells omit ~/.local/bin from PATH.
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

# Fail fast: huggingface-cli login is a no-op stub and leaves you unauthenticated.
"$PYTHON" - <<'EOF'
import os
import sys

from huggingface_hub import whoami
from huggingface_hub.errors import LocalTokenNotFoundError

if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    sys.exit(0)
try:
    info = whoami()
except LocalTokenNotFoundError:
    print(
        "Not logged in to Hugging Face.\n"
        "huggingface-cli login no longer works (deprecated stub).\n"
        "Run this by itself, then download:\n"
        "  ./login_hf.sh\n"
        "  # or: hf auth login\n"
        "  ./download_checkpoint.sh",
        file=sys.stderr,
    )
    sys.exit(2)
except Exception as err:
    print(
        f"Hugging Face auth failed: {err}\n"
        "Run: ./login_hf.sh   (or export HF_TOKEN=hf_...)",
        file=sys.stderr,
    )
    sys.exit(2)
name = info.get("name") if isinstance(info, dict) else info
print(f"Hugging Face user: {name}")
EOF

echo "Downloading facebook/sam3 (sam3.pt). This is several GB."
"$PYTHON" - <<'EOF'
import os
import shutil

from huggingface_hub import hf_hub_download

repo, ckpt, cfg = "facebook/sam3", "sam3.pt", "config.json"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
os.makedirs("checkpoints", exist_ok=True)
hf_hub_download(repo, cfg, token=token)
src = hf_hub_download(repo, ckpt, token=token)
dest = os.path.abspath(os.path.join("checkpoints", ckpt))
shutil.copy(src, dest)
print(f"Ready: {dest}")
EOF
