#!/bin/sh
# Start the module. Packaged builds use the PyInstaller binary; local
# `viam module reload` on the robot uses .venv after ./setup.sh.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Required for AMD GPUs not in PyTorch's official ROCm support list. NVIDIA needs
# nothing here: the CUDA build carries its own runtime and finds the driver itself.
if [ -d /opt/rocm ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

# Packaged GPU builds: onedir (dist/main/main). CPU/macOS: onefile (dist/main).
if [ -f "$SCRIPT_DIR/dist/main/main" ]; then
    exec "$SCRIPT_DIR/dist/main/main" "$@"
fi
if [ -f "$SCRIPT_DIR/dist/main" ]; then
    exec "$SCRIPT_DIR/dist/main" "$@"
fi

# Dev / local-module path (vino3 after ./setup.sh): run from the venv.
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
    cd "$SCRIPT_DIR/src" || exit 1
    exec "$SCRIPT_DIR/.venv/bin/python" main.py "$@"
fi

echo "sam3-detector: nothing to execute." >&2
echo "On the robot:  ./setup.sh && ./download_checkpoint.sh" >&2
echo "Packaged:      ./build.sh  (produces dist/main)" >&2
exit 1
