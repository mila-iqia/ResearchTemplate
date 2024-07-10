from .device import default_device

# Import this patch for https://github.com/mit-ll-responsible-ai/hydra-zen/issues/705 to make sure that it gets applied.
from .hydra_utils import patched_safe_name

__all__ = [
    "default_device",
    "patched_safe_name",
]
