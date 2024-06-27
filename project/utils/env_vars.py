import os
from pathlib import Path

import torch

SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_TMPDIR" in os.environ
    else tmp
    if "SLURM_JOB_ID" in os.environ and (tmp := Path("/tmp")).exists()
    else None
)
SLURM_JOB_ID: int | None = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)

NETWORK_DIR = (
    Path(os.environ["NETWORK_DIR"])
    if "NETWORK_DIR" in os.environ
    else _network_dir
    if (_network_dir := Path("/network")).exists()
    else None
)
"""The (read-only) network directory that contains datasets/weights/etc.

todo: adapt this for the DRAC clusters.

When running outside of the mila/DRAC clusters, this will be `None`, but can be mocked by setting the `NETWORK_DIR` environment variable.
"""

REPO_ROOTDIR = Path(__file__).parent
for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent

SCRATCH = Path(os.environ["SCRATCH"]) if "SCRATCH" in os.environ else None
"""SCRATCH directory where logs / checkpoints / custom datasets should be saved."""

DATA_DIR = Path(os.environ.get("DATA_DIR", (SLURM_TMPDIR or SCRATCH or REPO_ROOTDIR) / "data"))
"""Directory where datasets should be extracted."""


def get_constant(name: str):
    return globals()[name]


NUM_WORKERS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get(
            "SLURM_CPUS_ON_NODE",
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else torch.multiprocessing.cpu_count(),
        ),
    )
)
