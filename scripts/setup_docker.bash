#!/usr/bin/bash
mkdir -p $HOME/.local/bin
mkdir -p $HOME/.docker/cli-plugins

export DOCKER_VERSION=27.5.1
export DOCKER_BUILDX_VERSION=0.20.1

curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz | tar zxvf - --strip 1 -C $HOME/.local/bin docker/docker
curl -fsSL "https://github.com/docker/buildx-desktop/releases/download/v${DOCKER_BUILDX_VERSION}-desktop.2/buildx-v${DOCKER_BUILDX_VERSION}-desktop.2.linux-amd64" -o $HOME/.docker/cli-plugins/docker-buildx

chmod +x $HOME/.local/bin/docker $HOME/.docker/cli-plugins/docker-buildx


# NOTE: Then: `docker`
# echo "Get a token from the Github settings  -> Developer Settings page
# docker login ghcr.io
export DOCKER_HOST=tcp://127.0.0.1:2366
