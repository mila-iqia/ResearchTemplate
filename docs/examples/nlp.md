# NLP (PyTorch)


## Overview

The [HFExample][project.algorithms.hf_example.HFExample] is a [LightningModule][lightning.pytorch.core.module.LightningModule] for a simple auto-regressive text generation task.

It accepts a [HFDataModule][project.datamodules.text.HFDataModule] as input, along with a network.

??? note "Click to show the code for HFExample"
    {{ inline('project.algorithms.hf_example.HFExample', 4) }}

## Config files

### Algorithm config

??? note "Click to show the Algorithm config"
    Source: project/configs/algorithm/hf_example.yaml

    {{ inline('project/configs/algorithm/hf_example.yaml', 4) }}

### Datamodule config

??? note "Click to show the Datamodule config"
    Source: project/configs/datamodule/hf_text.yaml

    {{ inline('project/configs/datamodule/hf_text.yaml', 4) }}

## Running the example

Here is a configuration file that you can use to launch a simple experiment:

??? note "Click to show the yaml config file"
    Source: project/configs/experiment/hf_example.yaml

    {{ inline('project/configs/experiment/hf_example.yaml', 4) }}

You can use it like so:

```console
python project/main.py experiment=example
```
