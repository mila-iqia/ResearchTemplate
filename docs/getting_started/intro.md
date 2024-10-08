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

- ❗Support for both Jax and Torch with PyTorch-Lightning (See the [Jax example](../features/jax.md))❗
- Your Hydra configs will have an [Auto-Generated YAML schemas](../features/auto_schema.md) 🔥
- A comprehensive suite of automated tests for new algorithms, datasets and networks
    - 🤖 [Thoroughly tested on the Mila directly with GitHub CI](../features/testing.md#automated-testing-on-slurm-clusters-with-github-ci)
    - Automated testing on the DRAC clusters will also be added soon.
- Easy development inside a devcontainer with VsCode
- Tailor-made for ML researchers that run their jobs on SLURM clusters (with default configurations for the [Mila](https://docs.mila.quebec) and [DRAC](https://docs.alliancecan.ca) clusters.)
- Rich typing of all parts of the source code

This template is aimed for ML researchers that run their jobs on SLURM clusters.
The target audience is researchers and students at [Mila](https://mila.quebec). This template should still be useful for others outside of Mila that use PyTorch-Lightning and Hydra.
