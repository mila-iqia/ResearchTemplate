import dataclasses
import operator
from collections.abc import Callable
from typing import Concatenate, Literal, NamedTuple

import jax
import numpy as np
import torch
import torch.distributed
from lightning import Trainer
from torch_jax_interop import jax_to_torch, torch_to_jax

from project.algorithms.bases.algorithm import Algorithm
from project.datamodules.image_classification.base import ImageClassificationDataModule
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.types import PhaseStr

# type ParamsTuple = tuple[jax.Array, ...]


class ParamsTuple[T: torch.Tensor | jax.Array](NamedTuple):
    w1: T
    b1: T
    w2: T
    b2: T


def fcnet(input: jax.Array, params: ParamsTuple) -> jax.Array:
    """Forward pass of a simple two-layer fully-connected neural network with relu activation."""
    z1 = jax.numpy.matmul(input, params.w1) + params.b1
    a1 = jax.nn.relu(z1)
    logits = jax.numpy.matmul(a1, params.w2) + params.b2
    return logits


def loss_fn(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    assert isinstance(log_probs, jax.Array)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    assert isinstance(one_hot_labels, jax.Array)
    loss = -(one_hot_labels * log_probs).sum(axis=-1).mean()
    return loss


def forward_pass(
    params: ParamsTuple[jax.Array], x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    logits = fcnet(x, params)
    loss = loss_fn(logits, y)
    return loss, logits


def jit[**P, Out](
    fn: Callable[P, Out],
) -> Callable[P, Out]:
    """Small type hint fix for jax's `jit` (preserves the signature of the callable)."""
    return jax.jit(fn)  # type: ignore


def value_and_grad[In, **P, Out, Aux](
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Literal[0] = 0,
    has_aux: Literal[True] = True,
) -> Callable[Concatenate[In, P], tuple[tuple[Out, Aux], In]]:
    """Small type hint fix for jax's `value_and_grad` (preserves the signature of the callable)."""
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)  # type: ignore


# Register a handler for "converting" nn.Parameters to jax Arrays: they can be viewed as jax Arrays
# by just viewing their data as a jax array.
@torch_to_jax.register(torch.nn.Parameter)
def _parameter_to_jax_array(value: torch.nn.Parameter) -> jax.Array:
    return torch_to_jax(value.data)


class JaxAlgorithm(Algorithm):
    """Example of an algorithm where the forward / backward passes are written in Jax."""

    @dataclasses.dataclass
    class HParams(Algorithm.HParams):
        lr: float = 1e-3
        seed: int = 123
        debug: bool = False

    def __init__(
        self,
        *,
        datamodule: ImageClassificationDataModule,
        hp: HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, hp=hp or self.HParams())
        input_dims = int(np.prod(datamodule.dims))
        output_dims = datamodule.num_classes
        self.hp: JaxAlgorithm.HParams
        key = jax.random.key(self.hp.seed)
        # todo: Extract out the "network" portion, and probably use something like flax for it.
        params = ParamsTuple(
            w1=jax.random.uniform(key=jax.random.fold_in(key, 1), shape=(input_dims, 128)),
            b1=jax.random.uniform(key=jax.random.fold_in(key, 2), shape=(128,)),
            w2=jax.random.uniform(key=jax.random.fold_in(key, 3), shape=(128, output_dims)),
            b2=jax.random.uniform(key=jax.random.fold_in(key, 4), shape=(output_dims,)),
        )
        parameters, self.params_treedef = jax.tree.flatten(params)
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        self.params = torch.nn.ParameterList(
            map(operator.methodcaller("clone"), map(jax_to_torch, parameters))
        )

        # We will do the backward pass ourselves, and PL will only be used to synchronize stuff
        # between workers, do logging, etc.
        self.automatic_optimization = False

    def on_fit_start(self):
        # Setting those here, because otherwise we get pickling errors when running with multiple
        # GPUs.
        self.forward_pass = forward_pass
        self.backward_pass = value_and_grad(self.forward_pass)

        if not self.hp.debug:
            self.forward_pass = jit(self.forward_pass)
            self.backward_pass = jit(self.backward_pass)

    def jax_params(self) -> ParamsTuple[jax.Array]:
        # View the torch parameters as jax Arrays
        jax_parameters = jax.tree.map(torch_to_jax, list(self.params))
        # Reconstruct the original object structure.
        jax_params_tuple = jax.tree.unflatten(self.params_treedef, jax_parameters)
        return jax_params_tuple

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int, phase: PhaseStr
    ):
        x, y = batch
        # Note: flattening the input also gets rid of the stride issues in jax.from_dlpack.
        x = x.flatten(start_dim=1)
        # View/"convert" the torch inputs to jax Arrays.
        x, y = torch_to_jax(x), torch_to_jax(y)

        jax_params = self.jax_params()

        if phase != "train":
            # Only use the forward pass.
            loss, logits = self.forward_pass(jax_params, x, y)
        else:
            optimizer = self.optimizers()
            assert isinstance(optimizer, torch.optim.Optimizer)

            # Perform the backward pass
            (loss, logits), jax_grads = self.backward_pass(jax_params, x, y)
            distributed = torch.distributed.is_initialized()

            with torch.no_grad():
                # 'convert' the gradients to pytorch
                torch_grads = jax.tree.map(jax_to_torch, jax_grads)
                # Update the torch parameters tensors in-place using the jax grads.
                for param, grad in zip(self.parameters(), jax.tree.leaves(torch_grads)):
                    if distributed:
                        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG)
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad
            optimizer.step()
            optimizer.zero_grad()

        # IDEA: What about a hacky .backward method on a torch Tensor, that calls the backward pass
        # and sets the grads? Could we then use automatic optimization?
        torch_loss = jax_to_torch(loss)
        torch_y = batch[1]
        accuracy = jax_to_torch(logits).argmax(-1).eq(torch_y).float().mean()
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/loss", torch_loss, prog_bar=True, sync_dist=True)
        return torch_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)


def main():
    trainer = Trainer(devices="auto", accelerator="auto")
    datamodule = MNISTDataModule(num_workers=4)
    model = JaxAlgorithm(datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    ...


if __name__ == "__main__":
    main()
    print("Done!")
