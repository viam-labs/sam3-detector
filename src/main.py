import os
import sys

# Must be set before torch is imported anywhere — required for AMD ROCm GPUs.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import asyncio
from viam.module.module import Module
from models.sam3 import Sam3 as Sam3Model
from models.sam3_segments import Sam3Segments as Sam3SegmentsModel


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
