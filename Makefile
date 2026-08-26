# SAM3 Detector Module - Build
#
# Default checkpoint: facebook/sam3 (sam3.pt). Weights are gated on Hugging
# Face — request access, then ./login_hf.sh (or HF_TOKEN) before `make module`.
#
# `viam module reload` runs ./build.sh which packages SOURCE (not PyInstaller).
# first_run.sh on the robot then runs ./setup.sh.
#
# Usage:
#   make module                          # PyInstaller CUDA/ROCm/CPU bundle + tarball
#   make clean                           # Remove all build artifacts
#   make test                            # Unit tests (no weights required)
#
# The build target (GPU runtime and packaging mode) is detected by
# detect_target.sh. Override it with SAM3_BUILD_TARGET, e.g.:
#   make module SAM3_BUILD_TARGET=linux-cpu   # skip the ~5GB CUDA bundle

.PHONY: clean module test

clean:
	rm -rf dist/ build/
	rm -f module.tar.gz start

module:
	SAM3_MODEL=$(or $(SAM3_MODEL),facebook/sam3) SAM3_PACKAGE=pyinstaller ./build.sh

test:
	PYTHONPATH=src python -m pytest tests -q
