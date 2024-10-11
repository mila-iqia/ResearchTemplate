# Installation instructions

## Creating a new project from the template

To create a new project using this template, [_*Click Here*_](https://github.com/new?template_name=ResearchTemplate&template_owner=mila-iqia) or on the green "Use this template" button on [the template's GitHub repository](https://github.com/mila-iqia/ResearchTemplate).


## Setting up your development environment


1. Using [uv](https://docs.astral.sh/uv/)
2. Using a development container (recommended if you are able to install Docker on your machine)


=== "Locally (Linux / Mac)"

    1. Clone your new repo and navigate into it

    You can install the package using `pip install -e .`, but we recommend using [uv](https://docs.astral.sh/uv/)
    package manager. This makes it easier to switch python versions and to add or change the dependencies later on.

    2. On your machine:

        ```console
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.cargo/env
        uv sync      # Creates a virtual environment and installs dependencies in it.
        ```

=== "Locally (Windows)"

    1. Install WSL following [this guide](https://learn.microsoft.com/en-us/windows/wsl/install)
    2. Follow the installation instructions for Linux

=== "On a SLURM cluster"

    1. Clone your new repo and navigate into it

        ```bash
        git clone https://www.github.com/your-username/your-repo-name
        cd your-repo-name
        ```

    2. Launch the setup script

        If you're on the `mila` cluster, you can run the setup script on a *compute* node, just to be nice:

        ```console
        srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00 scripts/mila_setup.sh
        ```

## Using a development container

This repo provides a [Devcontainer](https://code.visualstudio.com/docs/remote/containers) configuration for [Visual Studio Code](https://code.visualstudio.com/) to use a Docker container as a pre-configured development environment. This avoids struggles setting up a development environment and makes them reproducible and consistent. Make yourself familiar with the [container tutorials](https://code.visualstudio.com/docs/remote/containers-tutorial) if you want to use them. In order to use GPUs, you can enable them within the `.devcontainer/devcontainer.json` file.

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
