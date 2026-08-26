#!/bin/sh
# Log in to Hugging Face. huggingface-cli is a deprecated stub in hub 1.28+
# and will NOT save a token — always use `hf auth login`.
set -e

export PATH="$HOME/.local/bin:$PATH"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -m pip install -U huggingface_hub
fi

HF_BIN=""
if [ -x "$HOME/.local/bin/hf" ]; then
    HF_BIN="$HOME/.local/bin/hf"
elif command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
fi

if [ -z "$HF_BIN" ]; then
    echo "ERROR: hf CLI not found. Install with:" >&2
    echo "  python3 -m pip install -U huggingface_hub" >&2
    exit 1
fi

echo "Create a Read token at https://huggingface.co/settings/tokens"
echo "Use the same Hugging Face account that was approved for facebook/sam3."
echo
echo "This is interactive — paste the token, then wait until login succeeds"
echo "before running ./download_checkpoint.sh"
echo
exec "$HF_BIN" auth login "$@"
