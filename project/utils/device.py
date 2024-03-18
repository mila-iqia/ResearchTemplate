import torch


def default_device() -> torch.device:
    """Returns the default device (GPU if available, else CPU)."""
    return torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
