# Import this patch for https://github.com/mit-ll-responsible-ai/hydra-zen/issues/705 to make sure that it gets applied.
from .utils import default_device

__all__ = [
    "default_device",
]
