"""Shared device selection, checkpoint lookup, and image helpers for SAM3 models."""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, Optional

import numpy as np
from PIL import Image as PILImage
from viam.logging import getLogger
from viam.media.video import ViamImage

from models.boxes import mask_to_bbox  # noqa: F401  — re-exported

LOGGER = getLogger(__name__)

# Official SAM 3 checkpoint. SAM 3.1 multiplex is a separate (incompatible)
# weight file; this module targets the public `facebook/sam3` image + video
# builders, which load `sam3.pt`.
SAM3_MODEL_ID = "facebook/sam3"
SAM3_CKPT_NAME = "sam3.pt"
SAM3_CFG_NAME = "config.json"

HF_ACCESS_URL = "https://huggingface.co/facebook/sam3"

GATED_CHECKPOINT_HELP = f"""
SAM 3 weights are gated on Hugging Face. Until Meta approves your request,
this module cannot load the detector — that is expected, not a bug.

After you get the approval email:
  1. huggingface-cli login
     (or export HF_TOKEN=hf_... with a token from https://huggingface.co/settings/tokens)
  2. From this repo, run:  ./download_checkpoint.sh
  3. Restart the Viam module. It will use checkpoints/{SAM3_CKPT_NAME}.

Request / status: {HF_ACCESS_URL}
""".strip()


class GatedCheckpointError(RuntimeError):
    """Raised when facebook/sam3 cannot be downloaded because access is pending."""


def is_gated_access_error(err: BaseException) -> bool:
    """True for Hugging Face 401/403 / gated-repo failures (pending approval)."""
    if type(err).__name__ == "GatedRepoError":
        return True
    text = str(err).lower()
    markers = (
        "gated",
        "cannot access gated",
        "agree to share",
        "401 client error",
        "403 client error",
        "401 unauthorized",
        "403 forbidden",
        "access to this model",
    )
    return any(m in text for m in markers)


def wrap_checkpoint_error(err: BaseException) -> BaseException:
    """Rewrite a raw HF download failure into an actionable error."""
    if is_gated_access_error(err):
        return GatedCheckpointError(f"{GATED_CHECKPOINT_HELP}\n\nOriginal error: {err}")
    return err


def torch_build_info() -> Dict[str, str]:
    """Describe the bundled torch build. Useful for diagnosing CPU fallback remotely."""
    import torch

    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        gpu_support = f"rocm-{hip_version}"
    elif torch.version.cuda:
        gpu_support = f"cuda-{torch.version.cuda}"
    else:
        gpu_support = "none (CPU-only build)"
    return {"torch_version": torch.__version__, "torch_gpu_support": gpu_support}


def _cpu_fallback_reason() -> str:
    """Explain why no GPU was selected, distinguishing a CPU-only build from a
    GPU-capable build that could not reach a device."""
    import torch

    if not torch.version.cuda and not getattr(torch.version, "hip", None):
        return (
            "this build of torch has no GPU support compiled in (CPU-only wheel), "
            "so no GPU can be used regardless of the installed hardware"
        )
    vendor = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    try:
        count = torch.cuda.device_count()
    except Exception as err:  # pragma: no cover - depends on driver state
        return f"{vendor} runtime is bundled but querying devices failed: {err}"
    if count == 0:
        return (
            f"{vendor} runtime is bundled but no device is visible — check that the "
            f"GPU driver is installed and loaded, that the module's user can access "
            f"the device nodes, and that CUDA_VISIBLE_DEVICES is not restricting it"
        )
    return f"{vendor} reports {count} device(s) but torch considers none usable"


def select_device() -> str:
    """Pick cuda, mps, or cpu. Logs the choice (warning on CPU fallback)."""
    import torch

    build = torch_build_info()
    if torch.cuda.is_available():
        # torch.cuda covers ROCm too; HIP builds report AMD cards through this API.
        LOGGER.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} "
            f"(torch {build['torch_version']}, {build['torch_gpu_support']})"
        )
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        LOGGER.info(f"Using Apple MPS (torch {build['torch_version']})")
        return "mps"
    LOGGER.warning(
        f"Falling back to CPU inference — {_cpu_fallback_reason()}. "
        f"(torch {build['torch_version']}, {build['torch_gpu_support']})"
    )
    return "cpu"


def checkpoint_search_dirs() -> list:
    """Directories that may contain a bundled `checkpoints/sam3.pt`."""
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),  # src/models/
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),  # repo root
        os.getcwd(),
    ]
    if hasattr(sys, "_MEIPASS"):
        search_dirs.insert(0, sys._MEIPASS)
    return search_dirs


def find_bundled_checkpoint() -> Optional[str]:
    """Return the path to a bundled sam3.pt, or None if it is not packaged."""
    for d in checkpoint_search_dirs():
        path = os.path.join(d, "checkpoints", SAM3_CKPT_NAME)
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def resolve_checkpoint_path() -> Optional[str]:
    """Locate a local checkpoint, or return None so the SAM3 builder will
    download from Hugging Face (`facebook/sam3`).

    SAM 3 weights are gated. If they are not bundled and Hugging Face auth is
    missing, model load will fail with a clear 401/403 from huggingface_hub.
    """
    bundled = find_bundled_checkpoint()
    if bundled:
        LOGGER.debug(f"Using bundled SAM3 checkpoint: {bundled}")
        return bundled
    LOGGER.info(
        f"No bundled {SAM3_CKPT_NAME}; SAM3 will download {SAM3_MODEL_ID} "
        f"from Hugging Face (requires an approved request + HF_TOKEN). "
        f"See {HF_ACCESS_URL}"
    )
    return None


def download_checkpoint(dest_dir: Optional[str] = None) -> str:
    """Download sam3.pt into dest_dir (default: <repo>/checkpoints).

    Raises GatedCheckpointError while the Hugging Face request is still pending.
    """
    from huggingface_hub import hf_hub_download

    if dest_dir is None:
        dest_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints"
        )
    os.makedirs(dest_dir, exist_ok=True)
    try:
        hf_hub_download(SAM3_MODEL_ID, SAM3_CFG_NAME)
        src = hf_hub_download(SAM3_MODEL_ID, SAM3_CKPT_NAME)
    except Exception as err:
        raise wrap_checkpoint_error(err) from err
    dest = os.path.abspath(os.path.join(dest_dir, SAM3_CKPT_NAME))
    import shutil

    shutil.copy(src, dest)
    LOGGER.info(f"Downloaded SAM3 checkpoint to {dest}")
    return dest


def viam_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a ViamImage to a numpy RGB array (H, W, 3)."""
    pil = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(pil)


def numpy_to_pil(image_np: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(image_np).convert("RGB")
