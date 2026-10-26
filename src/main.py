import os

# Must be set before torch / huggingface_hub / tqdm are imported.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import models.tqdm_silence  # noqa: F401  — patch tqdm before SAM3/HF Hub import

# isort: split

import asyncio

from viam.module.module import Module

import sdk_reconfigure
from models.sam3 import Sam3 as Sam3Model  # noqa: F401
from models.sam3_segments import Sam3Segments as Sam3SegmentsModel  # noqa: F401

if __name__ == "__main__":
    sdk_reconfigure.install()
    asyncio.run(Module.run_from_registry())
