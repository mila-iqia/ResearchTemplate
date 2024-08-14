#!/bin/bash
module purge
module load python/3.10
python -m venv ~/.venvs/venv
source ~/.venvs/venv/bin/activate
pip install pipx
pipx install pdm
deactivate

pdm install
