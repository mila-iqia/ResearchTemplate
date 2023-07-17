#!/bin/bash
set -e
set -x
set -v
module load anaconda/3
ENV_PATH="${SLURM_TMPDIR:?SLURM_TMPDIR is not set.}/env"
# NOTE: Getting lots of issues with Python 3.10, switching to python 3.9 instead.
conda create -y -p $ENV_PATH python=3.9 conda conda-libmamba-solver -c conda-forge
conda activate $ENV_PATH

export CONDA_EXE="$(hash -r; which conda)"
conda config --set solver libmamba
conda install -y -p $ENV_PATH conda-pack

# NOTE: pytorch-lightning==1.9.0 doesn't work (`import lightning` raises a ModuleNotFoundError)
# And there isn't a lightning==1.9 on conda-forge.

conda install -y -p $ENV_PATH -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision pytorch-cuda=11.7 hydra-core lightning==2.0.5 torchmetrics \
    wandb tqdm gdown pygame gym==0.26.1 hydra-submitit-launcher pandas matplotlib rich \
    pytest pytest-xdist pytest-timeout pytest-regressions
conda run -p $ENV_PATH pip install hydra-zen hydra-colorlog hydra-orion-sweeper

# <!-- conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 numba terminaltables matplotlib scikit-learn pandas assertpy pytz -c pytorch -c conda-forge
# pip install ffcv==0.0.3 -->
# conda create -n template -c pytorch -c nvidia -c conda-forge python=3.10 \
#     pytorch torchvision pytorch-cuda=11.7 hydra-core pytorch-lightning==1.9.0 torchmetrics wandb \
#     tqdm gdown pygame gym==0.26.1 hydra-submitit-launcher pandas matplotlib rich \
