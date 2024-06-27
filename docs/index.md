# Research Project Template

<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

![Build](https://github.com/mila-iqia/ResearchTemplate/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/mila-iqia/ResearchTemplate/graph/badge.svg?token=I2DYLK8NTD)](https://codecov.io/gh/mila-iqia/ResearchTemplate)

Please note: This is a Work-in-Progress. The goal is to make a first release by the end of summer 2024.

This is a research project template. It is meant to be a starting point for ML researchers at [Mila](https://mila.quebec/en).

Please follow the installation instructions [here](install.md)

## Overview

This project makes use of the following libraries:

- [Hydra](https://hydra.cc/) is used to configure the project. It allows you to define configuration files and override them from the command line.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is used to as the training framework. It provides a high-level interface to organize ML research code.
  - Please note: This repo does not restrict you to use PyTorch. You can also use Jax, as is shown in the [Jax example](https://www.github.com/mila-iqia/ResearchTemplate/tree/master/project/algorithms/jax_algo.py)
- [Weights & Biases](wandb.ai) is used to log metrics and visualize results.
- [pytest](https://docs.pytest.org/en/stable/) is used for testing.

## Usage

To see all available options:

```bash
python project/main.py --help
```

todo

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

<!--
## How does it work?

todo  -->

## Running tests

todo -->
