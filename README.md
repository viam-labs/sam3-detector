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

## Run on vino3 (NVIDIA Linux)

`viam module reload` (cloud) failed when `run.sh` was the entrypoint: Viam saw
that executable already in the repo, **skipped `./build.sh`**, uploaded no
`module.tar.gz`, and aborted. The `go.sum` cache lines are a red herring (this
is not a Go module).

Pull this fix, then from `~/viam/sam3-detector` on your laptop (or the robot):

```bash
cd ~/viam/sam3-detector
git pull
viam module reload --part-id=<part-id-of-the-linux-machine>
```

That cloud job now packages **source** (seconds, not a 5 GB PyInstaller bundle).
On the robot, `first_run.sh` then runs `./setup.sh` and downloads `sam3.pt` if
`HF_TOKEN` is set on the module. See [Hugging Face access](#hugging-face-access).

If the tree is **already** on the robot with a venv (skip the cloud builder):

```bash
cd ~/viam/sam3-detector
./setup.sh && ./login_hf.sh && ./download_checkpoint.sh
viam module reload-local --part-id=<part-id> --no-build
```

Or add a **local** module whose executable is `~/viam/sam3-detector/run.sh`
(see `local-module.example.json`) and restart that module in the app.

Do not copy a `.venv` from a Mac or from this cloud workspace onto vino3.

PyInstaller CUDA bundle (optional, for registry publish): `make module`.

## Hugging Face access

SAM 3 checkpoints are **gated**. Request access at
[facebook/sam3](https://huggingface.co/facebook/sam3) and wait for Meta's email.

`viam module reload` does **not** copy `~/.cache/huggingface` from your laptop.
The robot 401s unless you send a token or the weights. Do **one** of:

**1. Module env var (recommended)** — in the Viam app, on the `sam3-detector`
module card, add environment variable `HF_TOKEN` = a Hugging Face **Read**
token. `first_run.sh` and the detector both read it. See
`local-module.example.json`.

**2. `hf_token` file** — copy `hf_token.example` to `hf_token`, paste the
token, then reload. That file is **not** gitignored so reload copies it.

**3. Ship `checkpoints/sam3.pt`** — that directory is no longer gitignored.
Download once on the laptop (`./download_checkpoint.sh`), then reload so the
~3.3 GB file is in the module tarball.

Do **not** use `huggingface-cli login` — in huggingface_hub 1.28+ it is a
deprecated stub and will not save a token.

```bash
cd ~/viam/sam3-detector
./login_hf.sh                 # laptop only; paste a Read token
./download_checkpoint.sh      # optional if you want to ship sam3.pt
git pull
viam module reload --part-id=<part-id>
```

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

`meta.json` entrypoint is generated `start` (gitignored) so cloud reload
actually runs `./build.sh`. After `./setup.sh` that wrapper execs
`.venv/bin/python src/main.py`. After `make module` it prefers `dist/main`.

```bash
./run.sh                            # same launcher viam-server uses
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
SAM3_PACKAGE=pyinstaller SAM3_BUILD_TARGET=linux-cpu ./build.sh
```

## Testing against a robot

```bash
cd ~/viam/sam3-detector
export VIAM_API_KEY="..."
export VIAM_API_KEY_ID="..."
PYTHONPATH=src python test_camera.py --num-frames 20 --label "stemless wine glass"
PYTHONPATH=src python test_vision_service.py
```
