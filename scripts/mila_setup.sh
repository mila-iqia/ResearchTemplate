#!/bin/bash

# See https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425 for why this is useful.
set -euo pipefail

module purge
SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp}


# Check if we already installed pdm globally with pipx.
if [ ! -f ~/.local/bin/pdm.py ]; then
    module load python/3.10
    python -m venv $SLURM_TMPDIR/venv


    source $SLURM_TMPDIR/venv/bin/activate

    # Install pipx (so we can install pdm globally).
    pip install pipx

    # Install pdm globally, which creates an executable at ~/.local/bin/pdm
    pipx install --force pdm

    echo  "Making a pdm executable that includes the 'module load python/3.10' line. at ~/.local/bin/pdm"
    mv ~/.local/bin/pdm ~/.local/bin/pdm.py

    cat > ~/.local/bin/pdm << "EOF"
#!/bin/bash
module load python/3.10
~/.local/bin/pdm.py "$@"
EOF
    # Make this new `pdm` executable so we can call `pdm` from the command-line.
    chmod +x ~/.local/bin/pdm
    # Deactivate the virtual environment, we don't need it anymore.
    deactivate
    echo "Done setting up pdm."
fi


# Set the cache_dir of pdm in $SCRATCH (default is in $HOME/.cache/pdm which isn't great.)
mkdir -p $SCRATCH/.cache
pdm config cache_dir $SCRATCH/.cache/pdm


pdm install
pdm config
