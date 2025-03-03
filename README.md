<div align="center">

# Research Project Template

[![Build](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/mila-iqia/ResearchTemplate/graph/badge.svg?token=I2DYLK8NTD)](https://codecov.io/gh/mila-iqia/ResearchTemplate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mila-iqia/ResearchTemplate#license)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

🚀 Get started on a new research project with a clean, robust and well-tested base that you can count on! 🚀

</div>

This is a project template for ML researchers developed at [Mila](https://www.mila.quebec). Our goal with this is to help you get started with a new research project.

See [this introduction to the project](https://mila-iqia.github.io/ResearchTemplate/intro) for a detailed description of the context and motivations behind this project.

🚧 Please note: This is a work-in-progress and will get better over time! We want your feedback!🙏

## Installation

Projects created with this template use [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) to manage dependencies.
First, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) to install `uv` on your machine.

Once you have `uv` installed locally, use this command to create a new project from this template:

```bash
uvx copier copy --trust gh:mila-iqia/ResearchTemplate new_project
```

This will ask you a few questions to help set things up.
Navigate to the new directory, open up your favorite IDE, and voila! You're all setup! 🎊

Use this command to see all available options:

```bash
cd new_project  # replace with your folder.
. .venv/bin/activate
python project/main.py --help
```

For more detailed instructions, take a look at [this page](https://mila-iqia.github.io/ResearchTemplate/#setting-up-your-environment).

## Overview

This project makes use of the following libraries:

- [Hydra](https://hydra.cc/) is used to configure the project. It allows you to define configuration files and override them from the command line.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is used to as the training framework. It provides a high-level interface to organize ML research code.
    - 🔥 Please note: You can also use [Jax](https://jax.readthedocs.io/en/latest/) with this repo, as is shown in the [Jax example](https://mila-iqia.github.io/ResearchTemplate/examples/jax_image_classification/) 🔥
- [Weights & Biases](https://wandb.ai) is used to log metrics and visualize results.
- [pytest](https://docs.pytest.org/en/stable/) is used for testing.

## Who is this for? Why should you use this template?

This template comes with [some unique features that can *only* be found here (for now)!](https://mila-iqia.github.io/ResearchTemplate/features/)

- [Torch and Jax support](https://mila-iqia.github.io/ResearchTemplate/features/jax/)
- [Rich IDE support for Hydra config files](https://mila-iqia.github.io/ResearchTemplate/features/auto_schema/)
- [Built-in automated tests (including reproducibility tests), including testing on SLURM clusters!](https://mila-iqia.github.io/ResearchTemplate/features/testing/)
- And more! (see [this page](https://mila-iqia.github.io/ResearchTemplate/features/))

To make the best use of this template, you should ideally already have a good understanding of Python, some experience with PyTorch, and some basic experience with SLURM.

See [this page](https://mila-iqia.github.io/ResearchTemplate/resources/#other-project-templates) for a list of other templates to choose from if this isn't for you.

Please consider making an issue on this repo if you feel like this could be improved, or something is confusing to you. We very much need and appreciate your feedback! 😊

## Usage

To see all available options:

```bash
python project/main.py --help
```

For a detailed list of examples, see the [examples page](https://mila-iqia.github.io/ResearchTemplate/examples).

<!-- * `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit. -->

## Project layout

```
pyproject.toml   # Project metadata and dependencies
project/
    main.py      # main entry-point
    algorithms/  # learning algorithms
    datamodules/ # datasets, processing and loading
    networks/    # Neural networks used by algorithms
    configs/     # configuration files
docs/            # documentation
conftest.py      # Test fixtures and utilities
```
