import dataclasses
import operator
from collections.abc import Callable
from typing import Concatenate, Literal

import flax.linen
import jax
import optax
import torch
import torch.distributed
from flax.typing import VariableDict
from lightning import Trainer
from torch_jax_interop import jax_to_torch, torch_to_jax

from project.algorithms.bases.algorithm import Algorithm
from project.datamodules.image_classification.base import ImageClassificationDataModule
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.types import PhaseStr


class CNN(flax.linen.Module):
    """A simple CNN model.

    Taken from https://flax.readthedocs.io/en/latest/quick_start.html#define-network
    """

    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class FcNet(flax.linen.Module):
    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


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


def is_channels_first(shape: tuple[int, int, int] | tuple[int, int, int, int]) -> bool:
    if len(shape) == 4:
        return is_channels_first(shape[1:])
    return (shape[0] in (1, 3) and shape[1] not in {1, 3} and shape[2] not in {1, 3}) or (
        shape[0] < min(shape[1], shape[2])
    )


def to_channels_last[T: jax.Array | torch.Tensor](tensor: T) -> T:
    shape = tuple(tensor.shape)
    assert len(shape) == 3 or len(shape) == 4
    if not is_channels_first(shape):
        return tensor
    if isinstance(tensor, jax.Array):
        if len(shape) == 3:
            return tensor.transpose(1, 2, 0)
        return tensor.transpose(0, 2, 3, 1)
    else:
        if len(shape) == 3:
            return tensor.transpose(0, 2)
        return tensor.transpose(1, 3)


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
        network: flax.linen.Module,
        datamodule: ImageClassificationDataModule,
        hp: HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, hp=hp or self.HParams())
        self.hp: JaxAlgorithm.HParams
        key = jax.random.key(self.hp.seed)
        self.network = network
        x = jax.random.uniform(key, shape=(datamodule.batch_size, *datamodule.dims))
        x = to_channels_last(x)
        params = self.network.init(key, x=x)
        params_list, self.params_treedef = jax.tree.flatten(params)
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        self.params = torch.nn.ParameterList(
            map(operator.methodcaller("clone"), map(jax_to_torch, params_list))
        )

        # We will do the backward pass ourselves, and PL will only be used to synchronize stuff
        # between workers, do logging, etc.
        self.automatic_optimization = False

    def on_fit_start(self):
        # Setting those here, because otherwise we get pickling errors when running with multiple
        # GPUs.

        def loss_fn(params: VariableDict, x: jax.Array, y: jax.Array):
            logits = self.network.apply(params, x)
            assert isinstance(logits, jax.Array)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
            assert isinstance(loss, jax.Array)
            return loss, logits

        self.forward_pass = loss_fn
        self.backward_pass = value_and_grad(self.forward_pass)

        if not self.hp.debug:
            self.forward_pass = jit(self.forward_pass)
            self.backward_pass = jit(self.backward_pass)

    def jax_params(self) -> VariableDict:
        # View the torch parameters as jax Arrays
        jax_parameters = jax.tree.map(torch_to_jax, list(self.parameters()))
        # Reconstruct the original object structure.
        jax_params_tuple = jax.tree.unflatten(self.params_treedef, jax_parameters)
        return jax_params_tuple

    # def on_before_batch_transfer(
    #     self, batch: tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    # ):
    #     # Convert the batch to jax Arrays.
    #     x, y = batch
    #     # Seems like jax likes channels last tensors: jax.from_dlpack doesn't work with
    #     # channels-first tensors, so we have to do a transpose here.
    #     x = to_channels_last(x)
    #     # View the torch inputs as jax Arrays.
    #     x, y = torch_to_jax(x), torch_to_jax(y)
    #     return x, y

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int, phase: PhaseStr
    ):
        x, y = batch
        # Convert the batch to jax Arrays.
        # Seems like jax likes channels last tensors: jax.from_dlpack doesn't work with
        # channels-first tensors, so we have to do a transpose here.
        x = to_channels_last(x)
        # View the torch inputs as jax Arrays.
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
    model = JaxAlgorithm(network=CNN(num_classes=datamodule.num_classes), datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    ...


if __name__ == "__main__":
    main()
    print("Done!")
