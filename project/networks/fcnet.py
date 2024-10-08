"""An example of a simple fully connected network."""

from dataclasses import field

import numpy as np
import pydantic
from torch import nn


class FcNet(nn.Sequential):
    @pydantic.dataclasses.dataclass
    class HParams:
        """Dataclass containing the network hyper-parameters.

        This is an example of how Pydantic can be used to validate configs and command-line
        arguments.
        """

        hidden_dims: list[pydantic.PositiveInt] = field(default_factory=[128, 128].copy)

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
                block_layers.append(nn.Flatten())

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


# There's a bug in Hydra, it can't have a _target_ that is an inner class. Here we do a small trick
# for it to work.
HParams = FcNet.HParams
