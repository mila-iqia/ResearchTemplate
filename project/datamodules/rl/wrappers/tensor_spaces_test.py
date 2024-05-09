import random
from typing import Any

import pytest
import torch

from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete, TensorSpace
from project.utils.tensor_regression import TensorRegressionFixture


@pytest.fixture(
    params=[torch.float32, torch.float64, torch.int32, torch.int64], ids="dtype={}".format
)
def box_dtype(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[(), (3,), (3, 4), (3, 4, 5)], ids="shape={}".format)
def box_shape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture()
def box_space(seed: int, device: torch.device, box_dtype: torch.dtype, box_shape: tuple[int, ...]):
    return TensorBox(
        low=-1.0, high=1.0, shape=box_shape, dtype=box_dtype, seed=seed, device=device
    )


def test_box_sample(box_space: TensorBox, tensor_regression: TensorRegressionFixture):
    sample_1 = box_space.sample()
    sample_2 = box_space.sample()
    assert sample_1.device == box_space.device == box_space.device
    assert sample_1.dtype == box_space.dtype == box_space.dtype
    if box_space.dtype.is_floating_point:
        assert not sample_1.equal(sample_2)

    tensor_regression.check({"sample_1": sample_1, "sample_2": sample_2})

    if not box_space.dtype.is_floating_point:
        possible_values = set()
        for min_value, max_value in zip(box_space.low.flatten(), box_space.high.flatten()):
            possible_values.update(range(min_value, max_value + 1))

        samples = [box_space.sample().flatten() for _ in range(len(possible_values) * 10)]

        collected_values = set()
        for sample in samples:
            for value in sample.flatten():
                collected_values.add(value.item())

        assert collected_values <= possible_values
        assert (
            collected_values == possible_values
        )  # harder to make pass because curse of dimensionality.


def test_box_contains(box_space: TensorBox, seed: int):
    for _ in range(10):
        assert box_space.sample() in box_space

    assert box_space.low in box_space
    assert box_space.high in box_space
    assert (box_space.low + 0.5 * (box_space.high - box_space.low)).to(
        dtype=box_space.dtype
    ) in box_space
    gen = random.Random(seed)

    for _ in range(10):
        assert (box_space.low + gen.random() * (box_space.high - box_space.low)).to(
            dtype=box_space.dtype
        ) in box_space


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


@pytest.fixture(params=[0, -1, 1], ids="start={}".format)
def start(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def discrete_space(seed: int, start: int, device: torch.device):
    return TensorDiscrete(n=10, start=start, seed=seed, device=device)


def test_discrete_sample(
    discrete_space: TensorDiscrete, tensor_regression: TensorRegressionFixture
):
    sample_1 = discrete_space.sample()
    sample_2 = discrete_space.sample()
    assert sample_1.device == discrete_space.device == discrete_space.device
    assert sample_1.dtype == discrete_space.dtype == discrete_space.dtype
    # assert not sample_1.equal(sample_2)
    tensor_regression.check({"sample_1": sample_1, "sample_2": sample_2})

    samples = [discrete_space.sample() for _ in range(100)]
    assert set([s.item() for s in samples]) == set(
        range(discrete_space.start, discrete_space.start + discrete_space.n)
    )


def test_discrete_contains(discrete_space: TensorDiscrete, seed: int):
    for _ in range(10):
        assert discrete_space.sample() in discrete_space

    start = torch.as_tensor(
        discrete_space.start, device=discrete_space.device, dtype=discrete_space.dtype
    )
    n = torch.as_tensor(discrete_space.n, device=discrete_space.device, dtype=discrete_space.dtype)
    end = start + n
    assert start - 1 not in discrete_space
    assert start in discrete_space
    assert end - 1 in discrete_space
    assert end not in discrete_space
    rng = random.Random(seed)
    for _ in range(10):
        assert (
            start + torch.empty_like(n).fill_(rng.randint(0, discrete_space.n - 1))
            in discrete_space
        )


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
