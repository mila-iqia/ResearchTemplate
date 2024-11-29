---
additional_python_references:
  - project.algorithms.image_classifier
  - lightning.pytorch.core.module
---

# Supervised Learning (PyTorch)


## ImageClassifier

The `ImageClassifier` is a simple `LightningModule` for image classification.
It accepts a vision datamodule as input.

??? note "Click to show the code of the ImageClassifier class."
    {{ inline('project.algorithms.image_classifier.ImageClassifier', 4) }}

## Running the example

Here is a configuration file that you can use to launch a simple experiment:

??? note "Click to show the yaml config file"
    {{ inline('project/configs/experiment/example.yaml', 4) }}

You can use it like so:

```console
python project/main.py experiment=example
```
