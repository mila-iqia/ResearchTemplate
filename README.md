# Research Project Template

[![Build](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/mila-iqia/ResearchTemplate/graph/badge.svg?token=I2DYLK8NTD)](https://codecov.io/gh/mila-iqia/ResearchTemplate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mila-iqia/ResearchTemplate#license)

Please note: This is a Work-in-Progress. The goal is to make a first release by the end of summer 2024.

This is a template repository for a research project in machine learning. It is meant to be a starting point for new ML researchers that run jobs on SLURM clusters.
The main target audience is [Mila](https://mila.quebec/en) researchers and students, but this should still be useful to anyone that uses PyTorch-Lightning with Hydra.

For more context, see [this  introduction to the project.](https://mila-iqia.github.io/ResearchTemplate/intro).

## Overview

This project makes use of the following libraries:

- [Hydra](https://hydra.cc/) is used to configure the project. It allows you to define configuration files and override them from the command line.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is used to as the training framework. It provides a high-level interface to organize ML research code.
    - 🔥 Please note: You can also use [Jax](https://jax.readthedocs.io/en/latest/) with this repo, as is shown in the [Jax example](https://mila-iqia.github.io/ResearchTemplate/examples/jax) 🔥
- [Weights & Biases](https://wandb.ai) is used to log metrics and visualize results.
- [pytest](https://docs.pytest.org/en/stable/) is used for testing.

## Why use this template?

Why should you use this template (instead of another)?

Here are some of the advantages to using this template compared to [some of the other templates out there](https://mila-iqia.github.io/ResearchTemplate/related):

- ❗Can use both PyTorch and Jax with PyTorch-Lightning ❗
- Neat Configs thanks to [Auto-Generated YAML schemas](https://mila-iqia.github.io/ResearchTemplate/features/auto_schema)
- Easy development inside a [Development Container](https://code.visualstudio.com/docs/remote/containers) with [VsCode](https://code.visualstudio.com/)
- Tailor-made for ML researchers that run their jobs on SLURM clusters (with default configurations for the [Mila](https://docs.mila.quebec) and [DRAC](https://docs.alliancecan.ca) clusters.)
- Rich typing and documentation of all parts of the source code using Python 3.12's new type annotation syntax
- A comprehensive suite of automated tests for all algorithms, datasets and networks that are easy to reuse and extend
- Automatically creates Yaml Schemas for your Hydra config files (as soon as #7 is merged)

## Usage

To see all available options:

```bash
python project/main.py --help
```

For a detailed list of examples, see the [examples page](https://mila-iqia.github.io/ResearchTemplate/examples/examples).

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
