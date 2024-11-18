---
additional_python_references:
  - project.algorithms.jax_image_classifier
  - project.trainers.jax_trainer
---

# Jax + PyTorch-Lightning ⚡

## A LightningModule that trains a Jax network

The `JaxImageClassifier` algorithm uses a network which is a [flax.linen.Module](https://flax.readthedocs.io/en/latest/).
The network is wrapped with `torch_jax_interop.JaxFunction`, so that it can accept torch tensors as inputs, produces torch tensors as outputs, and the parameters are saved as as `torch.nn.Parameter`s (which use the same underlying memory as the jax arrays).
In this example, the loss function and optimizers are in PyTorch, while the network forward and backward passes are written in Jax.

The loss that is returned in the training step is used by Lightning in the usual way. The backward
pass uses Jax to calculate the gradients, and the weights are updated by a PyTorch optimizer.

!!! info
    You could also very well do both the forward **and** backward passes in Jax! To do this, [use the 'manual optimization' mode of PyTorch-Lightning](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html) and perform the parameter updates yourself. For the rest of Lightning to work, just make sure to store the parameters as torch.nn.Parameters. An example of how to do this will be added shortly.



!!! question "What about end-to-end training in Jax?"

    See the [Jax RL Example](../examples/jax_rl.md)! :smile:

### Jax Network

{{ inline('project.algorithms.jax_image_classifier.JaxCNN') }}

### Jax Algorithm

{{ inline('project.algorithms.jax_image_classifier.JaxImageClassifier') }}

### Configs

#### LightningModule config

{{ inline('project/configs/algorithm/jax_image_classifier.yaml') }}

## Running the example

```console
$ python project/main.py algorithm=jax_image_classifier network=jax_cnn datamodule=cifar10
```
