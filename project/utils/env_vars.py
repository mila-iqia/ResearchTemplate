import os
from pathlib import Path

import torch

SLURM_JOB_ID: int | None = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)
"""The value of the 'SLURM_JOB_ID' environment variable.

See https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_ID.
"""

SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_TMPDIR" in os.environ
    else tmp
    if SLURM_JOB_ID is not None and (tmp := Path("/tmp")).exists()
    else None
)
"""The SLURM temporary directory, the fastest storage available.

- Extract your dataset to this directory at the start of your job.
- Remember to move any files created here to $SCRATCH since everything gets deleted at the end of the job.

See https://docs.mila.quebec/Information.html#slurm-tmpdir for more information.
"""

SCRATCH = Path(os.environ["SCRATCH"]) if "SCRATCH" in os.environ else None
"""Network directory where temporary logs / checkpoints / custom datasets should be saved.

Note that this is temporary storage. Files that you wish to be saved long-term should be saved to the `ARCHIVE` directory.

See https://docs.mila.quebec/Information.html#scratch for more information.
"""

ARCHIVE = Path(os.environ["ARCHIVE"]) if "ARCHIVE" in os.environ else None
"""Network directory for long-term storage. Only accessible from the login or cpu-only compute
nodes.

See https://docs.mila.quebec/Information.html#archive for more information.
"""


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
"""The root directory of this repository on this machine."""

for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


DATA_DIR = Path(os.environ.get("DATA_DIR", (SLURM_TMPDIR or SCRATCH or REPO_ROOTDIR) / "data"))
"""Local Directory where datasets should be extracted on this machine."""


def get_constant(name: str):
    """Resolver for Hydra to get the value of a constant in this file."""
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
"""Default number of workers to be used by dataloaders, based on the number of CPUs and/or
tasks."""
