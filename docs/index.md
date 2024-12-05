# Research Project Template

[![Build](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/mila-iqia/ResearchTemplate/graph/badge.svg?token=I2DYLK8NTD)](https://codecov.io/gh/mila-iqia/ResearchTemplate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mila-iqia/ResearchTemplate#license)

!!! note "Work-in-Progress"
    Please note: This is a work-in-progress and will get better over time! We want your feedback!🙏

This is a research project template. It is meant to be a starting point for ML researchers at [Mila](https://mila.quebec/en).

For more context, see [this  introduction to the project.](intro.md).

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    [Get started quickly](#starting-a-new-project) with [a single installation script](#) and get up
    and running in minutes

    [:octicons-arrow-right-24: Getting started](#starting-a-new-project)

- :test_tube:{ .lg .middle } __Well-tested, robust codebase__

    ---

    Focus on your research! Let tests take care of detecting bugs and broken configs!

    [:octicons-arrow-right-24: Check out the included tests](features/testing.md)

- :material-lightning-bolt:{ .lg .middle } __Support for both PyTorch and Jax__

    ---

    You can use both PyTorch and Jax for your algorithms!
    ([Lightning](https://lightning.ai/docs/pytorch/stable/) handles the rest.)

    [:octicons-arrow-right-24: Check out the Jax example](features/jax.md)

- :fontawesome-solid-plane-departure:{ .lg .middle } __Ready-to-use examples__

    ---

    Includes examples for Supervised learning(1) and NLP 🤗, with unsupervised learning and RL coming soon.
    { .annotate }

    1. The source code for the example is available [here](https://github.com/mila-iqia/ResearchTemplate/blob/master/project/algorithms/example.py)

    [:octicons-arrow-right-24: Check out the examples here](examples/index.md)

<!--
-   :material-scale-balance:{ .lg .middle } __Open Source, MIT__

    ---

    Material for MkDocs is licensed under MIT and available on [GitHub]

    [:octicons-arrow-right-24: License](#) -->

</div>


## Starting a new project

To create a new project using this template, [_*Click Here*_](https://github.com/new?template_name=ResearchTemplate&template_owner=mila-iqia) or on the green "Use this template" button on [the template's GitHub repository](https://github.com/mila-iqia/ResearchTemplate).


## Setting up your environment

Here are two recommended ways to setup your development environment:

* Using the [uv](https://rye.astral.sh/) package manager
* Using a development container (recommended if you are able to install Docker on your machine)


=== "Locally (Linux / Mac)"

    1. Clone your new repo and navigate into it

        ```bash
        git clone https://www.github.com/your-username/your-repo-name
        cd your-repo-name
        ```

    2. Install the package manager

        ```bash
        # Install uv
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        ```

    3. Install dependencies

        ```bash
        uv sync  # Creates a virtual environment and installs dependencies in it.
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

    2. (Mila cluster) - Launch the setup script

        If you're on the `mila` cluster, you can run the setup script on a *compute* node, just to be nice:

        ```console
        srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00 scripts/mila_setup.sh
        ```


## Usage

To see all available options:

```bash
uv run python project/main.py --help
```

For a detailed list of examples, see the [examples page](examples/index.md).


## Developing inside a container (advanced)

This repo provides a [Devcontainer](https://code.visualstudio.com/docs/remote/containers) configuration for [Visual Studio Code](https://code.visualstudio.com/) to use a Docker container as a pre-configured development environment. This avoids struggles setting up a development environment and makes them reproducible and consistent.

If that sounds useful to you, we recommend you first make yourself familiar with the [container tutorials](https://code.visualstudio.com/docs/remote/containers-tutorial) if you want to use them. The devcontainer.json file assumes that you have a GPU locally by default. If not, you can simply comment out the "--gpus" flag in the `.devcontainer/devcontainer.json` file.


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
