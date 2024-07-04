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

archive="actions-runner-linux-x64-2.317.0.tar.gz"
# Look for the actions-runner archive on $SCRATCH first. Download it if it doesn't exist.
if [ ! -f "$SCRATCH/$archive" ]; then
    curl -o $SCRATCH/$archive \
        -L "https://github.com/actions/runner/releases/download/v2.317.0/$archive"
fi
# Make a symbolic link in SLURM_TMPDIR.
ln --symbolic $SCRATCH/$archive $SLURM_TMPDIR/$archive

cd $SLURM_TMPDIR

echo "9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d  $archive" | shasum -a 256 -c

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.317.0.tar.gz

# NOTE: Could use this to get a token programmatically!
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-an-organization

# Create the runner and configure it programmatically
# todo: seems like --emphemeral isn't supported?
./config.sh --url https://github.com/mila-iqia/ResearchTemplate --token $TOKEN \
    --unattended --replace --name $SLURM_CLUSTER_NAME --labels $SLURM_CLUSTER_NAME

# Launch the actions runner.
./run.sh
