# Using Jax with PyTorch-Lightning

You can use Jax for your dataloading, your network, or the learning algorithm, all while still benefiting from the nice stuff that comes from using PyTorch-Lightning.

**How does this work?**
Well, we use [torch-jax-interop](https://www.github.com/lebrice/torch_jax_interop), another package developed here at Mila, which allows easy interop between torch and jax code. See the readme on that repo for more details.

You can use Jax in your network or learning algorithm, for example in your forward / backward passes, to update parameters, etc. but not the training loop itself, since that is handled by the [lightning.Trainer][lightning.pytorch.trainer.trainer.Trainer].
There are lots of good reasons why you might want to let Lightning handle the training loop.
which are very well described [here](https://lightning.ai/docs/pytorch/stable/).

??? note "What about end-to-end training in Jax?"

    See the [Jax RL Example (coming soon!)](https://github.com/mila-iqia/ResearchTemplate/pull/55)


## `JaxExample`: a LightningModule that uses Jax

The [JaxExample][project.algorithms.jax_example.JaxExample] algorithm uses a network which is a [flax.linen.Module](https://flax.readthedocs.io/en/latest/).
The network is wrapped with `torch_jax_interop.JaxFunction`, so that it can accept torch tensors as inputs, produces torch tensors as outputs, and the parameters are saved as as `torch.nn.Parameter`s (which use the same underlying memory as the jax arrays).
In this example, the loss function and optimizers are in PyTorch, while the network forward and backward passes are written in Jax.

The loss that is returned in the training step is used by Lightning in the usual way. The backward
pass uses Jax to calculate the gradients, and the weights are updated by a PyTorch optimizer.

!!! note
    You could also very well do both the forward **and** backward passes in Jax! To do this, [use the 'manual optimization' mode of PyTorch-Lightning](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html) and perform the parameter updates yourself. For the rest of Lightning to work, just make sure to store the parameters as torch.nn.Parameters. An example of how to do this will be added shortly.

### Jax Network

{{ inline('project.algorithms.jax_example.CNN') }}

### Jax Algorithm

{{ inline('project.algorithms.jax_example.JaxExample') }}

### Configs

#### JaxExample algorithm config

{{ inline('project/configs/algorithm/jax_example.yaml') }}

## Running the example

```console
$ python project/main.py algorithm=jax_example network=jax_cnn datamodule=cifar10
```
