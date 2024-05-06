from typing import Any

import pytest
import torch

from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete, TensorSpace
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


@pytest.fixture
def discrete_space(seed: int, device: torch.device):
    return TensorDiscrete(n=10, start=0, seed=seed, device=device)


def test_discrete_sample(
    discrete_space: TensorDiscrete, tensor_regression: TensorRegressionFixture
):
    sample_1 = discrete_space.sample()
    sample_2 = discrete_space.sample()
    assert sample_1.device == discrete_space.device == discrete_space.device
    assert sample_1.dtype == discrete_space.dtype == discrete_space.dtype

    samples = [discrete_space.sample() for _ in range(10)]
    assert not all(s_i.equal(samples[0]) for s_i in samples)

    # assert not sample_1.equal(sample_2)
    tensor_regression.check({"sample_1": sample_1, "sample_2": sample_2})


def test_discrete_contains(discrete_space: TensorDiscrete):
    sample = discrete_space.sample()
    assert discrete_space.contains(sample)
    assert sample in discrete_space


@pytest.mark.parametrize("value", ["Bob", 0, 1, 0.1, 0.22])
def test_discrete_doesnt_contain(discrete_space: TensorDiscrete, value: Any):
    assert not discrete_space.contains(value)
    assert value not in discrete_space


def test_discrete_space_device(discrete_space: TensorSpace, device: torch.device):
    space = discrete_space
    assert space.device == device
    sample = space.sample()
    assert sample.device == device
    # Tricky, but this should be true: Prevent moving stuff between devices as much as possible.
    other_device = "cpu" if device.type == "cuda" else "cuda"
    assert sample.to(device=other_device) not in space
