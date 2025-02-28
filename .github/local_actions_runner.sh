#!/bin/bash
## This script can be used to launch a new self-hosted GitHub runner.
## It assumes that the SH_TOKEN environment variable contains a GitHub token
## that is used to authenticate with the GitHub API in order to allow launching a new runner.
set -euo pipefail
set -o errexit
set -o nounset

readonly repo="mila-iqia/ResearchTemplate"
readonly action_runner_version="2.317.0"
readonly expected_checksum_for_version="9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d"

# Check for required commands.
for cmd in curl tar uvx; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed."
        exit 1
    fi
done

if [ -z "${SH_TOKEN:-}" ]; then
    echo "Error: SH_TOKEN environment variable is not set."
    echo "This script requires the SH_TOKEN environment variable be set to a GitHub token with permissions to create new self-hosted runners for the current repository."
    echo "To create this token, Follow the docs here: "
    echo " - https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
    echo " - and click here to create the new token: https://github.com/settings/personal-access-tokens/new"
    echo "The fine-grained token must have the '"Administration" repository permissions (write)' scope."
    exit 1
fi

# If we're on a SLURM cluster, download the archive to SCRATCH, but use the SLURM_TMPDIR as the working directory.
# Otherwise, use $HOME/scratch as the working directory.
WORKDIR="${SCRATCH:-$HOME}/actions-runners/$repo"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Setting up self-hosted runner in $WORKDIR"
archive="actions-runner-linux-x64-$action_runner_version.tar.gz"

# Look for the actions-runner archive. Download it if it doesn't exist.
if [ ! -f "$archive" ]; then
    curl --fail -o "$archive" \
        -L "https://github.com/actions/runner/releases/download/v$action_runner_version/$archive"
fi

# Check the archive integrity.
echo "$expected_checksum_for_version  $archive" | shasum -a 256 -c
# Extract the installer
tar xzf $archive
# Use the GitHub API to get a temporary registration token for a new self-hosted runner.
# This requires you to be an admin of the repository and to have the $SH_TOKEN secret set to a
# github token with (ideally only) the appropriate permissions.
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-a-repository
# Example output:
# {
#   "token": "XXXXX",
#   "expires_at": "2020-01-22T12:13:35.123-08:00"
# }
# Uses `uvx python` to just get python. Assumes that `uv` is already installed.
TOKEN=`curl --fail -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $SH_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/$repo/actions/runners/registration-token | \
  uvx python -c "import sys, json; print(json.load(sys.stdin)['token'])"`


# Create the runner and configure it programmatically with the token we just got
# from the GitHub API.
# TODO: Reconfigure it if it doesn't already exist? Or only configure it once?
# TODO: use --ephemeral to run only one job and exit? Or set it up to keep running?
./config.sh --url https://github.com/$repo --token $TOKEN \
  --unattended --replace --labels self-hosted --ephemeral

# BUG: Seems weird that we'd have to export those ourselves. Shouldn't they be set already?
export GITHUB_ACTIONS="true"
export RUNNER_LABELS="self-hosted"

# Launch the actions runner.
exec ./run.sh
