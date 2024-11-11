# Text Classification ( + 🤗)

## Overview

The [TextClassificationExample][project.algorithms.text_classification_example.TextClassificationExample] is a [LightningModule][lightning.pytorch.core.module.LightningModule] for a simple text classification task.

It accepts a [TextClassificationDataModule][project.datamodules.text.TextClassificationDataModule] as input, along with a network.

??? note "Click to show the code for HFExample"
    {{ inline('project.algorithms.text_classification_example.TextClassificationExample', 4) }}

## Config files

### Algorithm config

??? note "Click to show the Algorithm config"
    Source: project/configs/algorithm/text_classification_example.yaml

    {{ inline('project/configs/algorithm/text_classification_example.yaml', 4) }}

### Datamodule config

??? note "Click to show the Datamodule config"
    Source: project/configs/datamodule/glue_cola.yaml

    {{ inline('project/configs/datamodule/glue_cola.yaml', 4) }}

## Running the example

Here is a configuration file that you can use to launch a simple experiment:

??? note "Click to show the yaml config file"
    Source: project/configs/experiment/text_classification_example.yaml

    {{ inline('project/configs/experiment/text_classification_example.yaml', 4) }}

You can use it like so:

```console
python project/main.py experiment=text_classification_example
```
