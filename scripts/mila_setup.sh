#!/bin/bash

# See https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425 for why this is useful.
set -euo pipefail
module --quiet purge

# todo: Setup ~/.cache as a symlink to $SCRATCH/.cache ?

SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp}

if ! [ -x "$(command -v uv)" ]; then
    echo "Installing UV (a Python package manager, see https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
else
    echo "✅ UV is already installed"
fi

# BUG: Seems like the installation script adds something to ~/.profile, but this file isn't being
# run, even when there isn't a ~/.user_profile or ~/.bash_profile file?
# Therefore I'm adding a line in ~/.bash_aliases instead.
if ! grep -q 'source "$HOME/.cargo/env"' ~/.bash_aliases; then
    echo "Adding a line with 'source \"\$HOME/.cargo/env\"' to ~/.bash_aliases"
    echo "# Adding the uv command to path (https://docs.astral.sh/uv/)" >> ~/.bash_aliases
    echo 'source "$HOME/.cargo/env"' >> ~/.bash_aliases
else
    echo "✅ ~/.bash_aliases already contains 'source \"\$HOME/.cargo/env\"'"
fi

if ! grep -q 'module load libffi OpenSSL' ~/.bash_aliases; then
    echo "Adding a line with 'module load libffi OpenSSL' to ~/.bash_aliases"
    echo "# Load the libffi and OpenSSL modules (not Python)" >> ~/.bash_aliases
    echo 'module load libffi OpenSSL' >> ~/.bash_aliases
else
    echo "✅ ~/.bash_aliases already contains 'module load libffi OpenSSL'"
fi


if ! grep -q 'export UV_LINK_MODE=${UV_LINK_MODE:-"symlink"}' ~/.bash_aliases; then
    echo "Adding a line with 'export UV_LINK_MODE=\${UV_LINK_MODE:-"symlink"}' to ~/.bash_aliases"
    echo '# Setting UV_LINK_MODE to symlink (so that uv can use a cache dir on $SCRATCH)' >> ~/.bash_aliases
    echo 'export UV_LINK_MODE=${UV_LINK_MODE:-"symlink"}' >> ~/.bash_aliases
else
    echo "✅ ~/.bash_aliases already contains 'export UV_LINK_MODE="symlink"'"
fi
export UV_LINK_MODE=${UV_LINK_MODE:-"symlink"}

# NOTE: This seems to fix issues with the Python extension of vscode not activating virtualenvs
# correctly.
# Check if this file exists. If not, create it with the following content:
# ~/.vscode-server/data/Machine/settings.json
# {"python.experiments.optOutFrom": ["pythonTerminalEnvVarActivation"]}
if [ ! -f "$HOME/.vscode-server/data/Machine/settings.json" ]; then
    echo "Creating $HOME/.vscode-server/data/Machine/settings.json"
    mkdir -p $HOME/.vscode-server/data/Machine
    echo '{"python.experiments.optOutFrom": ["pythonTerminalEnvVarActivation"]}' > "$HOME/.vscode-server/data/Machine/settings.json"
else
    echo "✅ $HOME/.vscode-server/data/Machine/settings.json already exists"
fi

if [ ! -f logs ]; then
    echo "Creating a logs symlink to $SCRATCH/logs"
    mkdir -p $SCRATCH/logs
    ln -s $SCRATCH/logs logs
else
    echo "'logs' directory already exists."
fi

# install all dependencies
echo "Installing all dependencies"
uv sync --all-extras --frozen

echo "🙌 All done! 🙌"
echo "🤖 Next, reload the vscode window (Ctrl+shift+P, then write 'reload window' and press Enter)"
echo "🐍 Then, make sure that VsCode uses your new virtual environment:"
echo "    - Open the VsCode Command palette (Ctrl+shift+P)"
echo "    - Write 'Python: Select Interpreter' and press Enter"
echo "    - Select the '.venv/bin/python' option, and press Enter."
echo "That should do it! "
echo "🙏 If you encounter any issues, please let us know here: 🙏"
echo "   https://github.com/mila-iqia/ResearchTemplate/issues/new/choose"
echo "🚀 Happy coding! 🚀"
