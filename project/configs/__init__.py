from __future__ import annotations

import os
from pathlib import Path

from hydra_plugins.custom_launcher.custom_launcher import (
    CustomSlurmLauncher,
    CustomSlurmQueueConf,
)

FILE = Path(__file__)
REPO_ROOTDIR = FILE.parent
for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
SLURM_JOB_ID: int | None = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)

from .algorithm import *  # noqa
from .config import Config
from .datamodule import *  # noqa

__all__ = [
    "Config",
    "SLURM_TMPDIR",
    "SLURM_JOB_ID",
    "REPO_ROOTDIR",
    "CustomSlurmLauncher",
    "CustomSlurmQueueConf",
]
