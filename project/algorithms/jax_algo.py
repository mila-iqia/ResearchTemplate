import dataclasses

import jax
import torch
from lightning import Trainer
from torch_jax_interop import jax_to_torch, torch_to_jax

from project.algorithms.bases.algorithm import Algorithm
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.types import PhaseStr, is_sequence_of
from project.utils.types.protocols import DataModule


def forward_pass(
    input: jax.Array, w1: jax.Array, b1: jax.Array, w2: jax.Array, b2: jax.Array
) -> jax.Array:
    """Forward pass of a simple two-layer fully-connected neural network with relu activation."""
    z1 = jax.numpy.matmul(input, w1) + b1
    a1 = jax.nn.relu(z1)
    z2 = jax.numpy.matmul(a1, w2) + b2
    return z2


def loss_fn(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    probs = jax.nn.log_softmax(logits)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    assert isinstance(one_hot_labels, jax.Array)
    assert isinstance(probs, jax.Array)
    loss = -(one_hot_labels * probs).sum(axis=-1).mean()
    return loss


class JaxAlgorithm(Algorithm):
    """Example of an algorithm where the forward / backward passes are written in Jax."""

    @dataclasses.dataclass()
    class HParams(Algorithm.HParams):
        lr: float = 1e-3
        seed: int = 123
        debug: bool = True

    def __init__(
        self,
        *,
        datamodule: DataModule | None = None,
        hp: HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, hp=hp or self.HParams())
        self.hp: JaxAlgorithm.HParams
        key = jax.random.key(self.hp.seed)
        self.w1 = torch.nn.Parameter(
            jax_to_torch(jax.random.uniform(key=jax.random.fold_in(key, 1), shape=(784, 128))),
            requires_grad=True,
        )
        self.b1 = torch.nn.Parameter(
            jax_to_torch(jax.random.uniform(key=jax.random.fold_in(key, 2), shape=(128,))),
            requires_grad=True,
        )
        self.w2 = torch.nn.Parameter(
            jax_to_torch(jax.random.uniform(key=jax.random.fold_in(key, 3), shape=(128, 10))),
            requires_grad=True,
        )
        self.b2 = torch.nn.Parameter(
            jax_to_torch(jax.random.uniform(key=jax.random.fold_in(key, 4), shape=(10,))),
            requires_grad=True,
        )
        self._forward = jax.jit(forward_pass) if not self.hp.debug else forward_pass

        # Get the gradients with respect to all parameters.
        # self._backward_pass = jax.value_and_grad(loss_fn, argnums=range(2, 6))

        self.automatic_optimization = False

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int, phase: PhaseStr
    ):
        torch_x, torch_y = batch
        # note: Also gets rid of the stride issues. in jax.from_dlpack.
        torch_x = torch_x.flatten(start_dim=1)
        jax_x, jax_y = jax.tree.map(torch_to_jax, [torch_x, torch_y])

        optimizer = self.optimizers()
        assert not isinstance(optimizer, list)
        if phase == "train":
            optimizer.zero_grad()

        torch_params = tuple(p.data for p in self.parameters())
        jax_params = jax.tree.map(torch_to_jax, torch_params)
        assert isinstance(jax_x, jax.Array)
        assert isinstance(jax_y, jax.Array)
        assert is_sequence_of(jax_params, jax.Array)

        jax_logits = self._forward(jax_x, *jax_params)
        jax_loss = loss_fn(jax_logits, jax_y)

        torch_loss = jax_to_torch(jax_loss)
        if phase == "train":

            def _loss_fn(params, x, y):
                logits = self._forward(x, *params)
                return loss_fn(logits, y)

            jax_grads = jax.grad(_loss_fn, 0)(jax_params, jax_x, jax_y)
            torch_grads = jax.tree.map(jax_to_torch, jax_grads)

            with torch.no_grad():
                for param, grad in zip(self.parameters(), torch_grads):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad

        if phase == "train":
            optimizer.step()
        torch_logits = jax_to_torch(jax_logits)
        accuracy = torch_logits.argmax(-1).eq(torch_y).float().mean()
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True)
        self.log(f"{phase}/loss", torch_loss, prog_bar=True)
        return torch_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)


def main():
    trainer = Trainer(devices=1, accelerator="auto")
    model = JaxAlgorithm()
    trainer.fit(model, datamodule=MNISTDataModule())

    ...


if __name__ == "__main__":
    main()
    print("Done!")
