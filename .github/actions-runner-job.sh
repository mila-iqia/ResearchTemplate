#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/runner_%j.out
#SBATCH --signal=B:TERM@60 # tells the controller to send SIGTERM to the job 1
                            # min before its time ends to give it a chance for
                            # better cleanup. If you cancel the job manually,
                            # make sure that you specify the signal as TERM like
                            # so `scancel --signal=TERM <jobid>`.
                            # https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/

set -euo pipefail

# module --quiet purge
# module load cuda/12.2.2



archive="actions-runner-linux-x64-2.317.0.tar.gz"
# Look for the actions-runner archive on $SCRATCH first. Download it if it doesn't exist.
if [ ! -f "$SCRATCH/$archive" ]; then
    curl -o $SCRATCH/$archive \
        -L "https://github.com/actions/runner/releases/download/v2.317.0/$archive"
fi
# Make a symbolic link in SLURM_TMPDIR.
ln --symbolic --force $SCRATCH/$archive $SLURM_TMPDIR/$archive

cd $SLURM_TMPDIR

echo "9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d  $archive" | shasum -a 256 -c

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.317.0.tar.gz

# NOTE: Could use this to get a token programmatically!
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-an-organization

# cluster=${SLURM_CLUSTER_NAME:-local}
cluster=${SLURM_CLUSTER_NAME:-`hostname`}

# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-a-repository
# curl -L \
#   -X POST \
#   -H "Accept: application/vnd.github+json" \
#   -H "Authorization: Bearer <YOUR-TOKEN>" \
#   -H "X-GitHub-Api-Version: 2022-11-28" \
#   https://api.github.com/repos/OWNER/REPO/actions/runners/registration-token

# Example output:
# {
#   "token": "XXXXX",
#   "expires_at": "2020-01-22T12:13:35.123-08:00"
# }


if ! command -v jq &> /dev/null; then
    echo "the jq command doesn't seem to be installed."

    if ! test -f ~/.local/bin/jq; then
        echo "jq is not found at ~/.local/bin/jq, downloading it."
        # TODO: this assumes that ~/.local/bin is in $PATH, I'm not 100% sure that this is standard.
        mkdir -p ~/.local/bin
        wget https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64 -O ~/.local/bin/jq
        chmod +x ~/.local/bin/jq
    fi
fi

source ~/.bash_aliases

TOKEN=`curl -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${SH_TOKEN:?The SH_TOKEN env variable is not set}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/mila-iqia/ResearchTemplate/actions/runners/registration-token | ~/.local/bin/jq -r .token`

# Create the runner and configure it programmatically
./config.sh --url https://github.com/mila-iqia/ResearchTemplate --token $TOKEN \
    --unattended --replace --name $cluster --labels $cluster $SLURM_JOB_ID --ephemeral

# Launch the actions runner.
./run.sh
