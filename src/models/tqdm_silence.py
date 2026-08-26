"""Disable tqdm progress bars before SAM3 (or any other library) imports tqdm.

SAM3 uses tqdm internally and the output goes to stderr, which Viam logs as
errors.
"""

import tqdm as _tqdm_module
import tqdm.auto as _tqdm_auto_module


class _SilentTqdm:
    """A no-op tqdm replacement that acts as an identity iterator."""

    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc=None, refresh=True):
        pass


_tqdm_module.tqdm = _SilentTqdm
_tqdm_auto_module.tqdm = _SilentTqdm
