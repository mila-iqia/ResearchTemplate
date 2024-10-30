# Hyper-Parameter Optimization

!!! note "Work-in-progress!"
    Please note that this is very much a work in progress!

This is a small example
Hydra and submitit make it very easy to launch lots of jobs on SLURM clusters.

hyper-parameter optimization (HPO)

## Hyper-Parameter Optimization with the Orion Hydra Sweeper

Here is a configuration file that you can use to launch a hyper-parameter optimization (HPO) sweep

??? note "Click to show the yaml config file"
    {{inline('project/configs/experiment/local_sweep_example.yaml', 4)}}

You can use it like so:

```console
python project/main.py experiment=local_sweep_example
```

## Hyper-Parameter Optimization on a SLURM cluster

??? note "Click to show the yaml config file"
    {{inline('project/configs/experiment/cluster_sweep_example.yaml', 4)}}

Here's how you can easily launch a sweep remotely on the Mila cluster.
If you are already on a slurm cluster, use the `"cluster=current"` config.

```console
python project/main.py experiment=cluster_sweep_example cluster=mila
```
