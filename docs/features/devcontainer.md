
# Developing inside a container (advanced)

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


## Launching container jobs on SLURM clusters

This part is still a work in progress. In principle, developing inside a devcontainer should make it easier to ship the images to slurm clusters and run them as jobs.
