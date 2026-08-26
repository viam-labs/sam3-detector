# SAM3 Detector Module - Build
#
# Default checkpoint: facebook/sam3 (sam3.pt). Weights are gated on Hugging
# Face — request access, then ./login_hf.sh (or HF_TOKEN) before `make module`.
#
# Local reload on NVIDIA Linux (vino3) does not need this Makefile:
#   ./setup.sh && ./download_checkpoint.sh
# then point a local Viam module at ./run.sh.
#
# Usage:
#   make module                          # Build binary, download checkpoint, create tarball
#   make clean                           # Remove all build artifacts
#   make test                            # Unit tests (no weights required)
#
# The build target (GPU runtime and packaging mode) is detected by
# detect_target.sh. Override it with SAM3_BUILD_TARGET, e.g.:
#   make module SAM3_BUILD_TARGET=linux-cpu   # skip the ~5GB CUDA bundle

.PHONY: clean module test

clean:
	rm -rf dist/ build/
	rm -f module.tar.gz

module:
	SAM3_MODEL=$(or $(SAM3_MODEL),facebook/sam3) ./build.sh

test:
	PYTHONPATH=src python -m pytest tests -q
