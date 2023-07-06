from __future__ import annotations
from torch import Tensor, nn
import torch
from dataclasses import dataclass, field
import numpy as np
import gym
import gym.spaces


def fcnet_for_env(
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Discrete | gym.spaces.Box,
    hparams: FcNet.HParams | None = None,
) -> FcNet:
    # NOTE: When we add support for VectorEnvs, this `env` should ideally be a single env.
    assert isinstance(observation_space, gym.spaces.Box)
    if isinstance(action_space, gym.spaces.Discrete):
        return FcNet(
            input_shape=observation_space.shape,
            output_shape=(action_space.n,),
            hparams=hparams,
        )
    else:
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1, "action space is assumed to be a vector of floats."
        return FcNet(
            input_shape=observation_space.shape,
            output_shape=(
                *action_space.shape[:-1],
                action_space.shape[-1] * 2,  # outputs for the action distribution mean and std.
            ),
            hparams=hparams,
        )


class Flatten(nn.Flatten):
    def forward(self, input: Tensor):
        # NOTE: The input Should have at least 2 dimensions for `nn.Flatten` to work, but it isn't
        # the case with a single observation from a single env.
        if input.ndim == 1:
            return input
        if input.is_nested:
            # NOTE: This makes 2d inputs 3d on purpose so they can be used with a nn.Flatten.
            return torch.nested.as_nested_tensor(
                [input_i.reshape([input_i.shape[0], -1]) for input_i in input.unbind()]
            )
        if input.ndim == 3:
            # FIXME: Hacky: don't collapse the `sequence length` dimension here.
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
        *,
        input_shape: tuple[int, ...] | int | None = None,
        output_shape: tuple[int, ...] | int | None = None,
        # (Optional, only temporarily there so all the networks can be created with the same
        # signature)
        in_channels: int | None = None,
        n_classes: int | None = None,
        hparams: HParams | None = None,
    ):
        if isinstance(input_shape, tuple):
            self.input_shape = input_shape
            self.input_dims = int(np.prod(input_shape))
        elif isinstance(input_shape, int):
            self.input_shape = (input_shape,)
            self.input_dims = input_shape
        else:
            assert input_shape is None
            self.input_shape = None
            self.input_dims = None

        if isinstance(output_shape, tuple):
            self.output_shape = output_shape
            self.output_dims = int(np.prod(output_shape))
        elif isinstance(output_shape, int):
            self.output_shape = (output_shape,)
            self.output_dims = output_shape
        elif n_classes is not None:
            self.output_shape = (n_classes,)
            self.output_dims = n_classes
        else:
            raise RuntimeError("Need to pass one of `output_shape` or `n_classes`.")
        self.hparams: FcNet.HParams = hparams or type(self).HParams()

        blocks = []
        for block_index, (in_dims, out_dims) in enumerate(
            zip(
                [self.input_dims, *self.hparams.hidden_dims],
                self.hparams.hidden_dims + [self.output_dims],
            )
        ):
            block_layers = []
            # if block_index == 0:
            #     block_layers.append(Flatten())

            if in_dims is None:
                block_layers.append(nn.LazyLinear(out_dims, bias=self.hparams.use_bias))
            else:
                block_layers.append(nn.Linear(in_dims, out_dims, bias=self.hparams.use_bias))

            if self.hparams.dropout_rate != 0.0:
                block_layers.append(nn.Dropout(p=self.hparams.dropout_rate))

            if block_index != len(self.hparams.hidden_dims):
                block_layers.append(self.hparams.activation_class())
            # else:
            # Last block. Reshape output to the desired shape.
            # block_layers.append(Reshape(self.output_shape))

            block = nn.Sequential(*block_layers)
            blocks.append(block)

        # blocks.append(
        #     nn.Sequential(
        #         Flatten(),
        #         nn.LazyLinear(self.hparams.hidden_dims[0]),
        #         nn.Dropout(p=0.5),
        #         nn.ReLU(),
        #     )
        # )
        # for h_prev, h in zip(self.hparams.hidden_dims[:-1], self.hparams.hidden_dims[1:]):
        #     blocks.append(
        #         nn.Sequential(
        #             nn.Linear(h_prev, h),
        #             nn.Dropout(p=0.5),
        #             nn.ReLU(),
        #         )
        #     )
        # # Last layer.
        # blocks.append(
        #     nn.Sequential(
        #         nn.Linear(self.hparams.hidden_dims[-1], self.output_dims),
        #         Reshape(self.output_shape),
        #     )
        # )

        super().__init__(*blocks)
