"""Silence tqdm without breaking Hugging Face Hub.

huggingface_hub 1.x does `from tqdm.auto import tqdm` then `tqdm.set_lock(...)`.
A dummy tqdm class without that classmethod makes Hub's lazy loader fail, which
Python reports as:

    ImportError: cannot import name 'hf_hub_download' from 'huggingface_hub'

SAM3's model_builder imports hf_hub_download at module level, so the vision
resource never starts.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from tqdm.std import tqdm as _RealTqdm


class _SilentTqdm(_RealTqdm):
    """Real tqdm subclass so Hub still finds set_lock/get_lock; bars stay off."""

    def __init__(self, iterable=None, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(iterable, *args, **kwargs)


import tqdm as _tqdm_module
import tqdm.auto as _tqdm_auto_module

_tqdm_module.tqdm = _SilentTqdm
_tqdm_auto_module.tqdm = _SilentTqdm
