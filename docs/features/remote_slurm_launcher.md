# Remote Slurm Submitit Launcher


This template includes a custom submitit launcher, that can be used to launch jobs on *remote* slurm clusters.
This allows you to develop code locally, and easily ship it to a different cluster.
The only prerequisite is that you must have `ssh` access to the remote cluster.

Under the hood, this uses a [custom `remote-slurm-executor` submitit plugin](https://github.com/lebrice/remote-slurm-executor).


This feature allows you to launch jobs on remote slurm clusters using two config groups:

- The `resources` config group is used to select the job resources:
    - `cpu`: CPU job
    - `gpu`: GPU job
- The `cluster` config group controls where to run the job:
    - `current`: Run on the current cluster. Assumes that you're already on a SLURM cluster (e.g. when using `mila code`). This uses the usual `submitit_slurm` launcher.
    - `mila`: Launches the job on the Mila cluster.
    - `narval`: Remotely launches the job on the Narval cluster
    - `cedar`: Remotely launches the job on the Cedar cluster
    - `beluga`: Remotely launches the job on the Beluga cluster


## Examples


### Remotely launching a GPU job on the Mila cluster from a local machine:
```bash
python project/main.py experiment=example resources=gpu cluster=mila
```

### Launching a job on a DRAC cluster from a local machine
```bash
python project/main.py experiment=example resources=gpu cluster=narval
```

### Launching a GPU job on Narval from the Mila cluster
Assuming you already have SSH access setup from `mila` to `narval`, the command is exactly the same as above:
```bash
python project/main.py experiment=example resources=gpu cluster=narval
```

### Launching a sweep while debugging code on the Mila cluster

```bash
python project/main.py experiment=example algorithm.optimizer.lr=0.01,0.02,0.03 resources=gpu
```


### Launching a GPU job on the Mila cluster from your laptop

```bash
python project/main.py experiment=example resources=gpu cluster=mila
```


### Launching a GPU job on the Narval cluster from your laptop

```bash
python project/main.py experiment=example resources=gpu cluster=narval
```

!!! warning

    NOTE: At the moment it is very important that the "cluster" config be used **after** the "resources" config! Otherwise the submitit_slurm launcher is used.
