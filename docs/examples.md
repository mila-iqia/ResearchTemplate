# Examples

## Simple run

```bash
python project/main.py algorithm=example_algo datamodule=mnist network=fcnet
```

## Running a Hyper-Parameter sweep on a SLURM cluster

```bash
python project/main.py experiment=cluster_sweep_example
```
