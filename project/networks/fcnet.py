from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatch

import numpy as np
import torch
from gym.spaces.utils import flatdim
from torch import Tensor, nn

from project.datamodules.bases.image_classification import ImageClassificationDataModule
from project.datamodules.rl.datamodule import RlDataModule
from project.utils.types.protocols import DataModule


class Flatten(nn.Flatten):
    def forward(self, input: Tensor):
        # NOTE: The input Should have at least 2 dimensions for `nn.Flatten` to work, but it isn't
        # the case with a single observation from a single env.
        if input.ndim <= 1:
            return input
        if input.is_nested:
            # NOTE: This makes 2d inputs 3d on purpose so they can be used with a nn.Flatten.
            return torch.nested.as_nested_tensor(
                [input_i.reshape([input_i.shape[0], -1]) for input_i in input.unbind()]
            )
        if input.ndim == 3:
            # FIXME: Hacky: don't collapse the `sequence length` dimension here.
            # TODO: Perhaps use a named dimension to detect this case?
            return input.reshape([input.shape[0], input.shape[1], -1])
        return super().forward(input)


class FcNet(nn.Sequential):
    @dataclass
    class HParams:
        """Dataclass containing the network hyper-parameters."""

        hidden_dims: list[int] = field(default_factory=[128, 128].copy)

        use_bias: bool = True

        dropout_rate: float = 0.5
        """Dropout rate.

        Set to 0 to disable dropout.
        """

        activation: str = "ReLU"

        @property
        def activation_class(self) -> type[nn.Module]:
            if hasattr(nn, self.activation):
                return getattr(nn, self.activation)
            # if hasattr(nn.functional, self.activation):
            #     return getattr(nn.functional, self.activation)
            for activation_name, activation_class in list(vars(nn).items()):
                if activation_name.lower() == self.activation.lower():
                    return activation_class
            raise ValueError(f"Unknown activation function: {self.activation}")

    def __init__(
        self,
        output_dims: int,
        input_shape: tuple[int, ...] | None = None,
        hparams: HParams | None = None,
    ):
        if input_shape:
            self.input_shape = input_shape
            self.input_dims = int(np.prod(input_shape))
        else:
            self.input_shape = None
            self.input_dims = None
        self.hparams: FcNet.HParams = hparams or type(self).HParams()

        self.output_dims = output_dims

        blocks: list[nn.Sequential] = []
        for block_index, (in_dims, out_dims) in enumerate(
            zip(
                [self.input_dims, *self.hparams.hidden_dims],
                self.hparams.hidden_dims + [self.output_dims],
            )
        ):
            block_layers = []
            if block_index == 0:
                block_layers.append(Flatten())

            if in_dims is None:
                block_layers.append(nn.LazyLinear(out_dims, bias=self.hparams.use_bias))
            else:
                block_layers.append(nn.Linear(in_dims, out_dims, bias=self.hparams.use_bias))

            if block_index != len(self.hparams.hidden_dims):
                if self.hparams.dropout_rate != 0.0:
                    block_layers.append(nn.Dropout(p=self.hparams.dropout_rate))
                block_layers.append(self.hparams.activation_class())

            block = nn.Sequential(*block_layers)
            blocks.append(block)

        super().__init__(*blocks)


@singledispatch
def make_fcnet_for(datamodule: DataModule, hparams: FcNet.HParams | None = None) -> FcNet:
    raise NotImplementedError(f"Unsupported datamodule type: {type(datamodule)}")


@make_fcnet_for.register(ImageClassificationDataModule)
def make_fcnet_for_vision(
    datamodule: ImageClassificationDataModule, hparams: FcNet.HParams | None = None
) -> FcNet:
    return FcNet(input_shape=datamodule.dims, output_dims=datamodule.num_classes, hparams=hparams)


@make_fcnet_for.register(RlDataModule)
def make_fcnet_for_rl(datamodule: RlDataModule, hparams: FcNet.HParams | None = None) -> FcNet:
    return FcNet(
        input_shape=(flatdim(datamodule.observation_space),),
        output_dims=flatdim(datamodule.action_space),
        hparams=hparams,
    )
