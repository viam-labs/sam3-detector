#!/bin/sh
# Load HF_TOKEN from the Viam module environment or a local hf_token file.
# Sourced by run.sh / first_run.sh / download_checkpoint.sh.
# The file is not gitignored so `viam module reload` copies it onto the robot.

_hf_token_file=""
if [ -n "${SAM3_ROOT:-}" ] && [ -f "$SAM3_ROOT/hf_token" ]; then
    _hf_token_file="$SAM3_ROOT/hf_token"
elif [ -f hf_token ]; then
    _hf_token_file="hf_token"
fi

if [ -z "$HF_TOKEN" ] && [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

if [ -z "$HF_TOKEN" ] && [ -n "$_hf_token_file" ]; then
    HF_TOKEN="$(grep -v '^[[:space:]]*#' "$_hf_token_file" | grep -v '^[[:space:]]*$' | head -1 | tr -d ' \t\r\n')"
fi

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
unset _hf_token_file
