---
additional_python_references:
  - project.algorithms.jax_rl_example
  - project.algorithms.example
  - project.algorithms.jax_example
  - project.algorithms.hf_example
  - project.trainers.jax_trainer
---

# Using Jax with PyTorch-Lightning

> 🔥 NOTE: This is a feature that is entirely unique to this template! 🔥

This template includes examples that use either Jax, PyTorch, or both!

| Example link                                      | Reference          | Framework   | Lightning?   |
| ------------------------------------------------- | ------------------ | ----------- | ------------ |
| [ExampleAlgorithm](../examples/jax_sl_example.md) | `ExampleAlgorithm` | Torch       | yes          |
| [JaxExample](../examples/jax_sl_example.md)       | `JaxExample`       | Torch + Jax | yes          |
| [HFExample](../examples/nlp.md)                   | `HFExample`        | Torch + 🤗   | yes          |
| [JaxRLExample](../examples/jax_rl_example.md)     | `JaxRLExample`     | Jax         | no (almost!) |


In fact, here you can mix and match both Jax and Torch code. For example, you can use Jax for your dataloading, your network, or the learning algorithm, all while still benefiting from the nice stuff that comes from using PyTorch-Lightning.

??? note "**How does this work?**"
    Well, we use [torch-jax-interop](https://www.github.com/lebrice/torch_jax_interop), another package developed here at Mila 😎, that allows easy interop between torch and jax code. Feel free to take a look at it if you'd like to use it as part of your own project. 😁



## Using PyTorch-Lightning to train a Jax network

If you'd like to use Jax in your network or learning algorithm, while keeping the same style of
training loop as usual, you can!

- Use Jax for the forward / backward passes, the parameter updates, dataset preprocessing, etc.
- Leave the training loop / callbacks / logging / checkpointing / etc to Lightning

The [lightning.Trainer][lightning.pytorch.trainer.trainer.Trainer] will not be able to tell that you're using Jax!

**Take a look at [this image classification example that uses a Jax network](../examples/jax_sl_example.md).**


## End-to-end training in Jax: the `JaxTrainer`

The `JaxTrainer`, used in the [Jax RL Example](../examples/jax_rl_example.md), follows a similar structure as the lightning Trainer. However, instead of training LightningModules, it trains `JaxModule`s, which are a simplified, jax-based look-alike of `lightning.LightningModule`s.


The "algorithm" needs to match the `JaxModule` protocol:
- `JaxModule.training_step`: train using a batch of data
