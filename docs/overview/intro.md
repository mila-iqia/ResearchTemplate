# Introduction

## Why should you use this template?

### Why should you use *a* template in the first place?

For many good reasons, which are very well described [here in a similar project](https://cookiecutter-data-science.drivendata.org/why/)! 😊

Other good reads:

- [https://cookiecutter-data-science.drivendata.org/why/](https://cookiecutter-data-science.drivendata.org/why/)
- [https://cookiecutter-data-science.drivendata.org/opinions/](https://cookiecutter-data-science.drivendata.org/opinions/)
- [https://12factor.net/](https://12factor.net/)
- [https://github.com/ashleve/lightning-hydra-template/tree/main?tab=readme-ov-file#main-ideas](https://github.com/ashleve/lightning-hydra-template/tree/main?tab=readme-ov-file#main-ideas)

### Why should you use *this* template (instead of another)?

You are welcome (and encouraged) to use other similar templates which, at the time of writing this, have significantly better documentation. However, there are several advantages to using this particular template:

- ❗Support for both Jax and Torch with PyTorch-Lightning ❗
- Easy development inside a devcontainer with VsCode
- Tailor-made for ML researchers that run their jobs on SLURM clusters (with default configurations for the [Mila](https://docs.mila.quebec) and [DRAC](https://docs.alliancecan.ca) clusters.)
- Rich typing of all parts of the source code using Python 3.12's new type annotation syntax
- A comprehensive suite of automated tests for new algorithms, datasets and networks
- Automatically creates Yaml Schemas for your Hydra config files (as soon as #7 is merged)

This template is geared specifically for ML researchers that run their jobs on SLURM clusters.
A particular emphasis  for development specifically with a SLURM cluster, and more particularly still, with the Mila and DRAC clusters in mind. The target audience is (currently) limited to Mila researchers, but there's no reason why this

## Main concepts

### Datamodule

### Network

### Algorithm
