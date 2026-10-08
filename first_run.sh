#!/bin/sh
# Runs once on the robot after install / `viam module reload`.
# Cloud reload ships source (not a PyInstaller bundle); this creates the
# platform venv and downloads gated sam3.pt using HF_TOKEN.
#
# The Hub download must happen here: viam-server only allows 2 minutes for
# resource configuration (VIAM_RESOURCE_CONFIGURATION_TIMEOUT), which is too
# short for a multi-GB checkpoint. first_run defaults to 1 hour.
set -e
cd "$(dirname "$0")"
SAM3_ROOT="$(pwd)"
# shellcheck disable=SC1091
. ./load_hf_env.sh

chmod +x run.sh setup.sh detect_target.sh download_checkpoint.sh login_hf.sh 2>/dev/null || true

write_start() {
    cat > start <<'EOF'
#!/bin/sh
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" "$@"
EOF
    chmod +x start
}

write_start

if [ -f dist/main/main ] || [ -f dist/main ]; then
    echo "Packaged binary present; skipping ./setup.sh"
else
    ./setup.sh
fi

if [ -f checkpoints/sam3.pt ]; then
    echo "Checkpoint already present: checkpoints/sam3.pt"
    exit 0
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: checkpoints/sam3.pt is missing and HF_TOKEN is not set." >&2
    echo "Set HF_TOKEN on the module (CONFIGURE → Environment) so first_run can" >&2
    echo "download facebook/sam3. Resource startup cannot download it (2-minute timeout)." >&2
    echo "Alternatively: ./login_hf.sh && ./download_checkpoint.sh, then reload." >&2
    exit 1
fi

# huggingface_hub's default HTTP timeouts are too short for a multi-GB file.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

echo "Downloading facebook/sam3 (sam3.pt) with HF_TOKEN during first_run..."
./download_checkpoint.sh
