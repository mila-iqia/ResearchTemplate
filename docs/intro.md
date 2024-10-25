# Why use this template?



## Why should you use *a* template in the first place?

For many good reasons, which are very well described [here in a similar project](https://cookiecutter-data-science.drivendata.org/why/)! 😊

Other good reads:

- [https://cookiecutter-data-science.drivendata.org/why/](https://cookiecutter-data-science.drivendata.org/why/)
- [https://cookiecutter-data-science.drivendata.org/opinions/](https://cookiecutter-data-science.drivendata.org/opinions/)
- [https://12factor.net/](https://12factor.net/)
- [https://github.com/ashleve/lightning-hydra-template/tree/main?tab=readme-ov-file#main-ideas](https://github.com/ashleve/lightning-hydra-template/tree/main?tab=readme-ov-file#main-ideas)

## Why use *this* template?

- [Cool, unique features that can *only* be found here (for now)!](features/index.md)


## Project layout

```
pyproject.toml   # Project metadata and dependencies
project/
    main.py      # main entry-point
    algorithms/  # learning algorithms
    datamodules/ # datasets, processing and loading
    networks/    # Neural networks used by algorithms
    configs/     # Hydra configuration files
docs/            # documentation
conftest.py      # Test fixtures and utilities
```

## Libraries used

This project makes use of the following libraries:

- [Hydra](https://hydra.cc/) is used to configure the project. It allows you to define configuration files and override them from the command line.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is used to as the training framework. It provides a high-level interface to organize ML research code.
    - 🔥 Please note: You can also use [Jax](https://jax.readthedocs.io/en/latest/) with this repo, as described in the [Jax example](features/jax.md) 🔥
- [Weights & Biases](https://wandb.ai) is used to log metrics and visualize results.
- [pytest](https://docs.pytest.org/en/stable/) is used for testing.
