# Supervised Learning (PyTorch)

The [ExampleAlgorithm][project.algorithms.ExampleAlgorithm] is a simple [LightningModule][lightning.pytorch.core.module.LightningModule] for image classification.

??? note "Click to show the code for ExampleAlgorithm"
    {{ inline('project.algorithms.example.ExampleAlgorithm', 4) }}

Here is a configuration file that you can use to launch a simple experiment:

??? note "Click to show the yaml config file"
    {{ inline('project/configs/experiment/example.yaml', 4) }}

You can use it like so:

```console
python project/main.py experiment=example
```
