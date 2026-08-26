#!/bin/sh
# Runs once on the robot after install / `viam module reload`.
# Cloud reload ships source (not a PyInstaller bundle); this creates the
# platform venv and tries to fetch gated sam3.pt.
set -e
cd "$(dirname "$0")"

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

if ./download_checkpoint.sh; then
    exit 0
fi

echo "WARNING: checkpoints/sam3.pt is missing." >&2
echo "On the robot:  ./login_hf.sh && ./download_checkpoint.sh" >&2
echo "Or set HF_TOKEN in the module environment and re-run first_run." >&2
# Do not fail: viam-server would refuse to start the module at all.
exit 0
