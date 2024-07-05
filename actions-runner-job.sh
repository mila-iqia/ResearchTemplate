#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus=rtx8000:1
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/runner_%j.out


set -eof pipefail

module --quiet purge
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
    # TODO: this assumes that ~/.local/bin is in $PATH, I'm not 100% sure that this is standard.
    echo "jq is not installed. Installing it."
    mkdir -p ~/.local/bin
    wget https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64 -O ~/.local/bin/jq
    chmod +x ~/.local/bin/jq
fi

TOKEN=`curl -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${SH_TOKEN:?Need to set the SH_TOKEN environment variable}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/mila-iqia/ResearchTemplate/actions/runners/registration-token | ~/local/bin/jq -r .token`


# Create the runner and configure it programmatically
./config.sh --url https://github.com/mila-iqia/ResearchTemplate --token $TOKEN \
    --unattended --replace --name $cluster --labels $cluster $SLURM_JOB_ID --ephemeral

# Launch the actions runner.
./run.sh
