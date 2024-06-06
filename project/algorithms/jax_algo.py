import dataclasses
import operator
import typing
from collections.abc import Callable
from typing import ClassVar, Concatenate, Literal

import flax.linen
import jax
import torch
import torch.distributed
from chex import PyTreeDef
from flax.typing import VariableDict
from lightning import Trainer
from torch_jax_interop import jax_to_torch, torch_to_jax

from project.algorithms.bases.algorithm import Algorithm
from project.datamodules.image_classification.base import ImageClassificationDataModule
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.types import PhaseStr, is_sequence_of


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))


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


class JaxOperation(torch.nn.Module):
    def __init__(
        self,
        jax_function: Callable[[VariableDict, jax.Array], jax.Array],
        jax_params_dict: VariableDict,
    ):
        super().__init__()
        self.jax_function = jax.jit(jax_function)
        params_list, self.params_treedef = jax.tree.flatten(jax_params_dict)
        # Register the parameters.
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        self.params = torch.nn.ParameterList(
            map(operator.methodcaller("clone"), map(jax_to_torch, params_list))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = JaxFunction.apply(
            input,
            self.params_treedef,
            self.jax_function,
            *self.params,
        )
        return out

    if typing.TYPE_CHECKING:
        __call__ = forward


class JaxFunction(torch.autograd.Function):
    """Wrapper for a jax function."""
    params_treedef: ClassVar

    @staticmethod
    def forward(
        ctx: torch.autograd.function.NestedIOFunction,
        input: torch.Tensor,
        params_treedef: PyTreeDef,
        jax_function: Callable[[VariableDict, jax.Array], jax.Array],
        *params: torch.Tensor,  # need to flatten the params for autograd to understand that they need a gradient.
    ):
        jax_input = torch_to_jax(input)
        jax_params = tuple(map(torch_to_jax, params))
        jax_params = jax.tree.unflatten(params_treedef, jax_params)

        needs_grad: tuple[bool, ...] = ctx.needs_input_grad  # type: ignore
        input_needs_grad, _, _, _, *params_need_grad = needs_grad
        if any(params_need_grad) or input_needs_grad:
            output, jvp_function =  jax.vjp(jax_function, jax_params, jax_input)
            ctx.jvp_function = jvp_function
        else:
            # Forward pass without gradient computation.
            output = jax_function(jax_params, jax_input)
        output = jax_to_torch(output)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        grad_output: torch.Tensor,
    ):
        input_need_grad,  _, _, *params_needs_grad = ctx.needs_input_grad
        # todo: broaden this a bit in case we need the grad of the input.
        # todo: Figure out how to do jax.grad for a function that outputs a matrix or vector.
        assert not input_need_grad
        grad_input = None
        if input_need_grad or any(params_needs_grad):
            assert all(params_needs_grad)
            jvp_function = ctx.jvp_function
            jax_grad_output = torch_to_jax(grad_output)
            jax_grad_params, jax_input_grad = jvp_function(jax_grad_output)
            params_grads = jax.tree.map(jax_to_torch, jax.tree.leaves(jax_grad_params))
            assert is_sequence_of(params_grads, torch.Tensor)

            if input_need_grad:
                grad_input = jax_to_torch(jax_input_grad)
        else:
            assert not any(params_needs_grad)
            params_grads = tuple(None for _ in params_needs_grad)
        return grad_input, None, None, *params_grads


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
        torch.zeros(1, device="cuda")  # weird cuda errors!
        key = jax.random.key(self.hp.seed)
        x = jax.random.uniform(key, shape=(datamodule.batch_size, *datamodule.dims))
        x = to_channels_last(x)
        jax_net = CNN()
        params = jax_net.init(key, x=x)
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.

        self.network = JaxOperation(jax_function=jax_net.apply, jax_params_dict=params)

        # self.params = torch.nn.ParameterList(
        #     map(operator.methodcaller("clone"), map(jax_to_torch, params_list))
        # )

        self.automatic_optimization = True

    def on_fit_start(self):
        pass
        # # Setting those here, because otherwise we get pickling errors when running with multiple
        # # GPUs.
        # def loss_function(
        #     params: VariableDict,
        #     x: jax.Array,
        #     y: jax.Array,
        # ):

        # self.forward_pass = loss_function
        # self.backward_pass = value_and_grad(self.forward_pass, argnums=0, has_aux=True)

        # if not self.hp.debug:
        #     self.forward_pass = jit(self.forward_pass)
        #     self.backward_pass = jit(self.backward_pass)

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int, phase: PhaseStr
    ):
        x, y = batch
        # Convert the batch to jax Arrays.
        # Seems like jax likes channels last tensors: jax.from_dlpack doesn't work with
        # channels-first tensors, so we have to do a transpose here.

        x = to_channels_last(x)

        logits = self.network(x)
        assert isinstance(logits, torch.Tensor)
        loss = torch.nn.functional.cross_entropy(logits, target=y).mean()
        assert isinstance(loss, torch.Tensor)
        if phase == "train":
            assert loss.requires_grad
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=True)
        acc = logits.argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)


def main():
    trainer = Trainer(devices=1, accelerator="auto")
    datamodule = MNISTDataModule(num_workers=4, batch_size=2)
    model = JaxAlgorithm(network=CNN(num_classes=datamodule.num_classes), datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    ...


if __name__ == "__main__":
    main()
    print("Done!")
