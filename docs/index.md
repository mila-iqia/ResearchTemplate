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

    [Get started quickly](#starting-a-new-project) with an interactive installation script and get up
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

<script src="https://asciinema.org/a/708495.js" id="asciicast-708495" async="true"></script>


## Setting up your environment


=== "Locally (Linux / Mac)"


    1. Install `uv`:

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    2. Create your new project

        Use this command to create a new project from this template:
        (Replace `research_project` with the path to the new project root folder.)

        ```bash
        uvx copier copy --trust gh:mila-iqia/ResearchTemplate research_project
        ```

        This will ask you a few questions and help setup your project.

=== "Locally (Windows)"

    1. Install WSL following [this guide](https://learn.microsoft.com/en-us/windows/wsl/install)
    2. Follow the installation instructions for Linux


Navigate to this new project, open up your favorite IDE, and voila! You're all setup! 🎊

Use this command to see all available options:

```bash
. .venv/bin/activate  # activate the virtual environment
python project/main.py --help
```




## Usage

To see all available options:

```bash
python project/main.py --help
```

For a detailed list of examples, see the [examples page](examples/index.md).
