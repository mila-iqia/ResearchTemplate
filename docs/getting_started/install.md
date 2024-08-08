# Installation instructions

There are two ways to install this project

1. Using Conda (recommended for newcomers)
2. Using a development container (recommended if you are able to install Docker on your machine)

## Installation

1. Clone the repository:

    ```bash
    git clone https://www.github.com/mila-iqia/ResearchTemplate
    cd ResearchTemplate
    ```

2. Installing dependencies

    You can install the package using `pip install -e .`, although we recommend using a package
    manager such as [PDM](https://pdm-project.org/en/latest/). This makes it easier to add or
    change the dependencies later on.

    1. On your machine:

        ```console
        pip install pdm
        pdm install       # Creates a virtual environment and installs dependencies in it.
        ```

    2. On the Mila cluster:

        If you're on the `mila` cluster, you can run this setup script (on a *compute* node):

        ```console
        salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00
        scripts/setup_mila.sh
        ```

## Using a development container

This repo provides a [Devcontainer](https://code.visualstudio.com/docs/remote/containers) configuration for [Visual Studio Code](https://code.visualstudio.com/) to use a Docker container as a pre-configured development environment. This avoids struggles setting up a development environment and makes them reproducible and consistent.  and make yourself familiar with the [container tutorials](https://code.visualstudio.com/docs/remote/containers-tutorial) if you want to use them. In order to use GPUs, you can enable them within the `.devcontainer/devcontainer.json` file.

1. Setup Docker on your local machine

    On an Linux machine where you have root access, you can install Docker using the following commands:

    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ```

    On Windows or Mac, follow [these installation instructions](https://code.visualstudio.com/docs/remote/containers#_installation)

2. (optional) Install the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to use your local machine's GPU(s).

3. Install the [Dev Containers extension](vscode:extension/ms-vscode-remote.remote-containers) for Visual Studio Code.

4. When opening repository in Visual Studio Code, you should be prompted to reopen the repository in a container:

    ![VsCode popup image](https://github.com/mila-iqia/ResearchTemplate/assets/13387299/37d00ce7-1214-44b2-b1d6-411ee286999f)

    Alternatively, you can open the command palette (Ctrl+Shift+P) and select `Dev Containers: Rebuild and Reopen in Container`.
