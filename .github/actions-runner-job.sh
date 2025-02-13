#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/runner_%j.out


set -euo pipefail

# todo: load modules here? or in the job steps?
# module --quiet purge
# module load cuda/12.0


archive="actions-runner-linux-x64-2.317.0.tar.gz"
# Look for the actions-runner archive on $SCRATCH first. Download it if it doesn't exist.
if [ ! -f "$SCRATCH/$archive" ]; then
    curl -o $SCRATCH/$archive \
        -L "https://github.com/actions/runner/releases/download/v2.317.0/$archive"
fi
# Make a symbolic link in SLURM_TMPDIR.
ln --symbolic --force $SCRATCH/$archive $SLURM_TMPDIR/$archive

cd $SLURM_TMPDIR

# Check the archive integrity.
echo "9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d  $archive" | shasum -a 256 -c

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.317.0.tar.gz

# Use the GitHub API to get a registration token for a self-hosted runner.
# This requires you to be an admin of the repository and to have the $SH_TOKEN secret set to your
# github token.
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-a-repository
# Example output:
# {
#   "token": "XXXXX",
#   "expires_at": "2020-01-22T12:13:35.123-08:00"
# }
source ~/.bash_aliases
module load python/3.10

TOKEN=`curl -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $SH_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/mila-iqia/ResearchTemplate/actions/runners/registration-token | \
  python -c "import sys, json; print(json.load(sys.stdin)['token'])"`

# Create the runner and configure it programmatically with the token we just got from the GitHub API.
cluster=$SLURM_CLUSTER_NAME
./config.sh --url https://github.com/mila-iqia/ResearchTemplate --token $TOKEN \
    --unattended --replace --name $cluster --labels $cluster $SLURM_JOB_ID --ephemeral

# Launch the actions runner.
./run.sh
