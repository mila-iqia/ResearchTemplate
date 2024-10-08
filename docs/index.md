# Research Project Template

[![Build](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/mila-iqia/ResearchTemplate/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/mila-iqia/ResearchTemplate/graph/badge.svg?token=I2DYLK8NTD)](https://codecov.io/gh/mila-iqia/ResearchTemplate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mila-iqia/ResearchTemplate#license)

!!! note "Work-in-Progress"
    Please note: This is a Work-in-Progress. The goal is to make a first release by the end of summer 2024.

This is a research project template. It is meant to be a starting point for ML researchers at [Mila](https://mila.quebec/en).

For more context, see [this  introduction to the project.](getting_started/intro.md).

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    [Get started quickly](getting_started/install.md) with [a single installation script](#) and get up
    and running in minutes

    [:octicons-arrow-right-24: Getting started](getting_started/install.md)

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

## Overview

This project makes use of the following libraries:

- [Hydra](https://hydra.cc/) is used to configure the project. It allows you to define configuration files and override them from the command line.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is used to as the training framework. It provides a high-level interface to organize ML research code.
    - 🔥 Please note: You can also use [Jax](https://jax.readthedocs.io/en/latest/) with this repo, as described in the [Jax example](features/jax.md) 🔥
- [Weights & Biases](https://wandb.ai) is used to log metrics and visualize results.
- [pytest](https://docs.pytest.org/en/stable/) is used for testing.

## Usage

To see all available options:

```bash
python project/main.py --help
```

For a detailed list of examples, see the [examples page](examples/index.md).

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
