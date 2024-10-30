# Remote Slurm Submitit Launcher

> 🔥 NOTE: This is a feature that is entirely unique to this template! 🔥

This template includes a custom submitit launcher, that can be used to launch jobs on *remote* slurm clusters.
This allows you to develop code locally, and easily ship it to a different cluster.
The only prerequisite is that you must have `ssh` access to the remote cluster.

Under the hood, this uses a [custom `remote-slurm-executor` submitit plugin](https://github.com/lebrice/remote-slurm-executor).


This feature allows you to launch jobs on remote slurm clusters using two config groups:

- The `resources` config group is used to select the job resources:
    - `cpu`: CPU job
    - `gpu`: GPU job
- The `cluster` config group controls where to run the job:
    - `current`: Run on the current cluster. Use this if you're already on a SLURM cluster (e.g. when using `mila code`). This uses the usual `submitit_slurm` launcher.
    - `mila`: Launches the job on the Mila cluster.
    - `narval`: Remotely launches the job on the Narval cluster
    - `cedar`: Remotely launches the job on the Cedar cluster
    - `beluga`: Remotely launches the job on the Beluga cluster


## Examples

This assumes that you've already setup SSH access to the clusters (for example using `mila init`).


### Local machine -> Mila

```bash
python project/main.py experiment=example resources=gpu cluster=mila
```

### Local machine -> DRAC (narval)

```bash
python project/main.py experiment=example resources=gpu cluster=narval
```


### Mila -> DRAC (narval)

This assumes that you've already setup SSH access from `mila` to the DRAC clusters.

Note that command is exactly the same as above.

```bash
python project/main.py experiment=example resources=gpu cluster=narval
```


!!! warning

    If you want to launch jobs on a remote cluster, it is (currently) necessary to place the "resources" config **before** the "cluster" config on the command-line.


## Launching jobs on the current SLURM cluster

If you develop on a SLURM cluster, you can use the `cluster=current`, or simply omit the `cluster` config group and only use a config from the `resources` group.

```bash
(mila) $ python project/main.py experiment=example resources=gpu cluster=current
```
