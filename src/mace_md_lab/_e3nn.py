"""Load e3nn's bundled Wigner constants with PyTorch's weights-only loader."""

from contextlib import nullcontext

import torch

# Older e3nn distributions serialize slice objects in their static constants.
# Scope the additional allowlisted type to the e3nn import.
allowed = getattr(torch.serialization, "safe_globals", None)
with allowed([slice]) if allowed else nullcontext():
    from e3nn import o3

__all__ = ["o3"]
