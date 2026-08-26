# SAM3 Detector Module

A Viam vision service module powered by Meta's [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3). This is a SAM2-detector-shaped module with the v2→v3 changes: **open-vocabulary text prompts**, so you can set a label like `"stemless wine glass"` without drawing a bounding box first.

Two models:

- **`viam:sam3-detector:sam3`** — Detect (and on CUDA, track) every instance of a text prompt
- **`viam:sam3-detector:sam3-segments`** — Same text prompt → SAM3 masks → depth projection → 3D point clouds

## What changed from SAM2

| | SAM2 detector | SAM3 detector |
|---|---|---|
| Prompt | Click / box (`initial_point_x/y`) required to start | **Text `label` required**; no box needed |
| Typical call | `set_point` then track one object | `set_label` `"stemless wine glass"`; all matching instances |
| Segments input | Upstream 2D detector bounding boxes | SAM3 text prompt (optional upstream filter) |
| GPU packaging | CUDA / ROCm / CPU via `detect_target.sh` | Same path (`SAM3_BUILD_TARGET`) |

## Models

### `viam:sam3-detector:sam3` — Text-prompted detection

Set `label` to a short noun phrase. SAM3 returns a bounding box and score for every matching instance. On NVIDIA/ROCm CUDA the module also runs the SAM3 video predictor over a sliding window of recent frames.

See [sam3 model documentation](viam_sam3-detector_sam3.md) for configuration details.

### `viam:sam3-detector:sam3-segments` — 3D segmentation

Uses the same text prompt to get precise masks, then projects only those pixels through the RGBD depth map. Point clouds are automatically transformed to the world frame using the machine's frame system.

See [sam3-segments model documentation](viam_sam3-detector_sam3-segments.md) for configuration details.

## Hugging Face access

SAM 3 checkpoints are **gated**. Submitting a request at
[facebook/sam3](https://huggingface.co/facebook/sam3) and waiting for Meta's
email is the normal path — the detector cannot run until that lands.

Clone this repo onto the machine that will run the module (for example
`~/viam/sam3-detector`). The tree is not created automatically on the robot.

**After the approval email**, on that machine, login first and wait until it
succeeds. Do not chain login and download on one line — `huggingface-cli login`
is a deprecated stub and will skip auth.

```bash
export PATH="$HOME/.local/bin:$PATH"
cd ~/viam/sam3-detector
hf auth login                 # paste a Read token, wait for "Login successful"
./download_checkpoint.sh      # copies sam3.pt into checkpoints/
```

If `hf` is not on `PATH` after install, use:

```bash
python3 -m huggingface_hub.cli.hf auth login
# or:
export HF_TOKEN=hf_...
./download_checkpoint.sh
```

Then start (or restart) the Viam module. If `checkpoints/sam3.pt` is missing it
will try Hugging Face again at first start and raise a clear error if access is
still pending.

Unit tests do not need the weights:

```bash
cd ~/viam/sam3-detector
PYTHONPATH=src python -m pytest tests -q
```

## Device selection

Both models auto-detect the best available device at startup:

| Platform | Device | Notes |
|---|---|---|
| Linux x86_64 + NVIDIA GPU | `cuda` | Published `linux/amd64` build; needs only an NVIDIA driver >= 525 |
| Linux + AMD GPU | `cuda` | ROCm build; see [Linux GPU setup](docs/linux-gpu-setup.md) |
| macOS Apple Silicon | `mps` | Metal Performance Shaders (per-frame image path) |
| Other | `cpu` | Fallback; per-frame image path |

Whichever device is chosen is logged at startup. If the module falls back to CPU it
logs a warning explaining why, and `do_command({"command": "status"})` reports the
selected `device` alongside `torch_version` and `torch_gpu_support`, so you can tell
a driver problem apart from a build that has no GPU support at all.

A GPU can only be used if the bundled PyTorch was built for it — a CPU-only wheel
will never use a GPU no matter what hardware is present. The published
`linux/amd64` build therefore ships CUDA-enabled PyTorch.

## Local development

```bash
cd ~/viam/sam3-detector
./setup.sh                          # venv + platform torch + deps
PYTHONPATH=src python -m pytest tests -q
```

Run as a local Viam module with `run.sh` after `./build.sh`, or from source:

```bash
cd src
../.venv/bin/python main.py
```

## Build targets

`detect_target.sh` picks a target from the build machine; override it with the
`SAM3_BUILD_TARGET` environment variable.

| Target | Selected on | PyTorch wheels | Packaging |
|---|---|---|---|
| `linux-cuda` | Linux x86_64 | CUDA 12.8 (`cu128`) | onedir |
| `linux-rocm` | Linux with `/opt/rocm` | ROCm 6.3 | onedir |
| `linux-cpu` | other Linux (e.g. arm64) | CPU-only | onefile |
| `darwin` | macOS | standard (MPS) | onefile |

The CUDA target bundles NVIDIA's CUDA runtime libraries, which makes it far larger
than the CPU build (roughly 5 GB unpacked). That is what lets the module run on a
machine that has only an NVIDIA driver and no CUDA toolkit installed. To build the
small CPU-only variant instead:

```bash
SAM3_BUILD_TARGET=linux-cpu ./build.sh
```

## Testing against a robot

```bash
cd ~/viam/sam3-detector
export VIAM_API_KEY="..."
export VIAM_API_KEY_ID="..."
PYTHONPATH=src python test_camera.py --num-frames 20 --label "stemless wine glass"
PYTHONPATH=src python test_vision_service.py
```
