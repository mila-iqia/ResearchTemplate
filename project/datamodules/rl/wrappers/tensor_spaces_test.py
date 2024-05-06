from typing import Any

import pytest
import torch

from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorSpace
from project.utils.tensor_regression import TensorRegressionFixture


@pytest.fixture
def box_space(seed: int, device: torch.device):
    return TensorBox(low=-1.0, high=1.0, shape=(3, 4), seed=seed, device=device)


def test_box_sample(box_space: TensorBox, tensor_regression: TensorRegressionFixture):
    sample_1 = box_space.sample()
    sample_2 = box_space.sample()
    assert sample_1.device == box_space.device == box_space.device
    assert sample_1.dtype == box_space.dtype == box_space.dtype
    assert not sample_1.equal(sample_2)
    tensor_regression.check({"sample_1": sample_1, "sample_2": sample_2})


def test_box_contains(box_space: TensorBox):
    sample = box_space.sample()
    assert box_space.contains(sample)
    assert sample in box_space


@pytest.mark.parametrize("value", ["Bob", 0, 1, 0.1, 0.22])
def test_box_doesnt_contain(box_space: TensorBox, value: Any):
    assert not box_space.contains(value)
    assert value not in box_space


def test_box_space_device(box_space: TensorSpace, device: torch.device):
    space = box_space
    assert space.device == device
    sample = space.sample()
    assert sample.device == device
    # Tricky, but this should be true: Prevent moving stuff between devices as much as possible.
    other_device = "cpu" if device.type == "cuda" else "cuda"
    assert sample.to(device=other_device) not in space
