#!/bin/sh
# Log in to Hugging Face using the CLI even when ~/.local/bin is not on PATH.
set -e

export PATH="$HOME/.local/bin:$PATH"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -m pip install -U huggingface_hub
fi

echo "Create a token at https://huggingface.co/settings/tokens (Read access is enough)."
echo "Use the same Hugging Face account that was approved for facebook/sam3."
echo

if [ -x "$HOME/.local/bin/hf" ]; then
    exec "$HOME/.local/bin/hf" auth login "$@"
fi
if [ -x "$HOME/.local/bin/huggingface-cli" ]; then
    exec "$HOME/.local/bin/huggingface-cli" login "$@"
fi
exec python3 -c "from huggingface_hub import interpreter_login; interpreter_login()"
