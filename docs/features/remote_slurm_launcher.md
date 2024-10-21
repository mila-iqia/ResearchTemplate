# Remote Slurm Submitit Launcher


This template includes a custom submitit launcher, that can be used to launch jobs on *remote* slurm clusters.
This allows you to develop code locally, and easily ship it to a different cluster.
The only prerequisite is that you must have `ssh` access to the remote cluster.


## the `cluster` config group


## The `resources` config group



## Examples

### Launching a GPU job on the Mila cluster from your laptop

```bash
python project/main.py experiment=example cluster=mila resources=one_gpu
```


### Launching a GPU job on the Narval cluster from your laptop

```bash
python project/main.py experiment=example cluster=narval resources=one_gpu
```



### Launching a GPU job on Narval from the Mila cluster

!!! note
    This assumes that you have setup ssh access from the mila cluster to Narval.


```bash
python project/main.py experiment=example cluster=narval resources=one_gpu
```
