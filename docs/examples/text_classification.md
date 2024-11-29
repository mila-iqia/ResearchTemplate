---
additional_python_references:
  - project.algorithms.text_classifier
  - project.datamodules.text.text_classification
---

# Text Classification (⚡ + 🤗)

## Overview

The `TextClassifier` is a [LightningModule][lightning.pytorch.core.module.LightningModule] for a simple text classification task.

It accepts a `TextClassificationDataModule` as input, along with a network.

??? note "Click to show the code of the lightningmodule"
    {{ inline('project.algorithms.text_classifier.TextClassifier', 4) }}

## Config files

### Algorithm config

??? note "Click to show the Algorithm config"
    Source: project/configs/algorithm/text_classifier.yaml

    {{ inline('project/configs/algorithm/text_classifier.yaml', 4) }}

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
