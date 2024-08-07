# Examples

TODOs:

- [ ] Show examples (that are also to be tested with doctest or similar) of how to add a new algo.
- [ ] Show examples of how to add a new datamodule.
- [ ] Add a link to the RL example once [#13](https://github.com/mila-iqia/ResearchTemplate/issues/13) is done.
- [ ] Add a link to the NLP example once [#14](https://github.com/mila-iqia/ResearchTemplate/issues/14) is done.
- [ ] Add an example of how to use Jax for the dataset/dataloading:
    - Either through an RL example, or with `tfds` in [#18](https://github.com/mila-iqia/ResearchTemplate/issues/18)

## Simple run

```bash
python project/main.py algorithm=example datamodule=mnist network=fcnet
```

## Running a Hyper-Parameter sweep on a SLURM cluster

```bash
python project/main.py experiment=cluster_sweep_example
```
