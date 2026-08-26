import os

# Must be set before torch / huggingface_hub / tqdm are imported.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import models.tqdm_silence  # noqa: F401  — patch tqdm before SAM3/HF Hub import

import asyncio
from viam.module.module import Module
from models.sam3 import Sam3 as Sam3Model
from models.sam3_segments import Sam3Segments as Sam3SegmentsModel


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
