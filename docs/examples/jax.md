# Using Jax

You can use Jax for your dataloading, your network, or the learning algorithm, all while still benefiting from the nice stuff that comes from using PyTorch-Lightning.

How does this work?
Well, we use [torch-jax-interop](https://www.github.com/lebrice/torch_jax_interop), another package developed here at Mila, which allows easy interop between torch and jax code. See the readme on that repo for more details.

## Example Algorithm that uses Jax

You can use Jax for your training step, but not the entire training loop (since that is handled by Lightning).
There are a few good reasons why you should let Lightning handle the training loop, most notably the fact that it handles all the logging, checkpointing, and other stuff that you'd lose if you swapped out the entire training framework for something based on Jax.

In this [example Jax algorithm](https://www.github.com/mila-iqia/ResearchTemplate/tree/master/project/algorithms/jax_algo.py),
a Neural network written in Jax (using [flax](https://flax.readthedocs.io/en/latest/)) is wrapped using the `torch_jax_interop.JaxFunction`, so that its parameters are learnable. The parameters are saved on the LightningModule as nn.Parameters (which use the same underlying memory as the jax arrays). In this example, the loss function is written in PyTorch, while the network forward and backward passes are written in Jax.

## Example datamodule that uses Jax

(todo)
