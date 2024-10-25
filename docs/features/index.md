
# Features unique to this project template

Here are some cool features that are unique to this particular template:


- Support for both Jax and Torch with PyTorch-Lightning (See the [Jax example](jax.md))
- Your Hydra configs will have an [Auto-Generated YAML schemas](auto_schema.md) 🔥
- A comprehensive suite of automated tests for new algorithms, datasets and networks
    - 🤖 [Thoroughly tested on the Mila directly with GitHub CI](testing.md#automated-testing-on-slurm-clusters-with-github-ci)
    - Automated testing on the DRAC clusters will also be added soon.
- Easy development inside a [devcontainer with VsCode]
- Tailor-made for ML researchers that run their jobs on SLURM clusters (with default configurations for the [Mila](https://docs.mila.quebec) and [DRAC](https://docs.alliancecan.ca) clusters.)
- Rich typing of all parts of the source code

This template is aimed for ML researchers that run their jobs on SLURM clusters.
The target audience is researchers and students at [Mila](https://mila.quebec). This template should still be useful for others outside of Mila that use PyTorch-Lightning and Hydra.
