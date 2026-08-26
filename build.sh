#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM3_MODEL="${SAM3_MODEL:-facebook/sam3}"
SAM3_CKPT="${SAM3_CKPT:-sam3.pt}"

# Default: source tarball for `viam module reload` (fast). PyInstaller CUDA
# bundles are opt-in — they are huge and the Viam cloud reload path does not
# have a Hugging Face token for sam3.pt.
#   ./build.sh                  # source + generated `start` entrypoint
#   SAM3_PACKAGE=pyinstaller ./build.sh
#   ./build.sh --bundle
PACKAGE="${SAM3_PACKAGE:-source}"
for arg in "$@"; do
    case "$arg" in
        --bundle|--pyinstaller) PACKAGE=pyinstaller ;;
        --source) PACKAGE=source ;;
    esac
done

write_start() {
    # meta.json entrypoint. Must not be committed: Viam cloud reload treats an
    # executable entrypoint already in the repo as a pre-built Go binary, skips
    # ./build.sh, and then fails because /tmp/module.tar.gz does not exist.
    cat > start <<'EOF'
#!/bin/sh
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" "$@"
EOF
    chmod +x start
}

write_start

if [ "$PACKAGE" = "source" ]; then
    echo "Packaging source tarball for viam module reload (no PyInstaller)."
    extra=""
    if [ -d checkpoints ]; then
        extra="$extra checkpoints"
        echo "Including checkpoints/ ($(du -sh checkpoints | cut -f1))"
    else
        echo "WARNING: checkpoints/ missing; robot will need HF_TOKEN or hf_token to download sam3.pt"
    fi
    if [ -f hf_token ]; then
        extra="$extra hf_token"
        echo "Including hf_token (Hugging Face credentials for gated sam3.pt)"
    fi
    # shellcheck disable=SC2086
    tar --exclude='__pycache__' --exclude='*.pyc' -czf module.tar.gz \
        meta.json run.sh start first_run.sh load_hf_env.sh \
        setup.sh detect_target.sh download_checkpoint.sh login_hf.sh \
        requirements.txt pyproject.toml .python-version \
        src \
        viam_sam3-detector_sam3.md viam_sam3-detector_sam3-segments.md \
        $extra
    echo "Built module.tar.gz ($(du -h module.tar.gz | cut -f1))"
    echo "On the robot, first_run.sh runs ./setup.sh and tries to download sam3.pt."
    exit 0
fi

# Resolve the target once and export it, so setup.sh and main.spec cannot
# disagree about which GPU runtime is being installed versus packaged.
. ./detect_target.sh
SAM3_BUILD_TARGET="$(detect_sam3_target)"
export SAM3_BUILD_TARGET
# main.spec also reads SAM2_BUILD_TARGET as a fallback alias.
export SAM2_BUILD_TARGET="$SAM3_BUILD_TARGET"
echo "Building PyInstaller bundle for target: $SAM3_BUILD_TARGET"

# Ensure dependencies are installed (creates venv, installs correct torch).
./setup.sh

# Use the venv python directly — never uv run/uv sync, which would
# re-resolve torch from PyPI and overwrite the GPU-enabled version.
PYTHON=".venv/bin/python"

# Fail early if the installed torch does not match the target, rather than
# shipping a CPU-only binary to a GPU platform.
$PYTHON - "$SAM3_BUILD_TARGET" <<'EOF'
import sys
import torch

target = sys.argv[1]
if getattr(torch.version, "hip", None):
    flavor = "linux-rocm"
elif torch.version.cuda:
    flavor = "linux-cuda"
else:
    flavor = "cpu-or-mps"
print(f"Bundling torch {torch.__version__} (GPU support: {flavor})")

if target in ("linux-cuda", "linux-rocm") and flavor != target:
    raise SystemExit(
        f"ERROR: target {target} needs a matching torch build, but the venv has "
        f"{flavor} ({torch.__version__}). Delete .venv and re-run, or check the "
        f"index URL in setup.sh."
    )
EOF

# Build PyInstaller binary using spec file (includes GPU runtime hooks).
$PYTHON -m PyInstaller --clean main.spec

# Download the model checkpoint if Hugging Face access is available. SAM 3
# weights are gated — without an accepted request + token the download fails
# and we still package the binary; the module will retry at runtime.
mkdir -p checkpoints
set +e
PYTHONPATH=src $PYTHON - <<'EOF'
from models.common import download_checkpoint
print(download_checkpoint("checkpoints"))
EOF
ckpt_status=$?
set -e
if [ "$ckpt_status" -ne 0 ]; then
    echo "WARNING: could not download ${SAM3_CKPT} from Hugging Face."
    echo "This is expected while your facebook/sam3 access request is pending."
    echo "After approval, run these separately (do not use huggingface-cli login):"
    echo "  ./login_hf.sh && ./download_checkpoint.sh"
fi

# Package into the tarball. dist/main is a directory (onedir/GPU builds) or a
# single file (onefile/CPU and macOS builds); run.sh handles both layouts.
# Quiet tar: a GPU bundle lists thousands of files, which buries build errors.
tar -czf module.tar.gz meta.json run.sh start first_run.sh load_hf_env.sh dist/main checkpoints/

echo "Built module.tar.gz ($(du -h module.tar.gz | cut -f1) packaged, $(du -sh dist/main | cut -f1) unpacked)"
