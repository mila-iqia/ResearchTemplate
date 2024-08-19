import time

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from project.datamodules.rl.wrappers.jax_torch_interop import jax_to_torch, torch_to_jax


@pytest.mark.timeout(10)
def test_jax_to_torch_benchmark(device: torch.device, benchmark: BenchmarkFixture):
    def _back_and_forth_loop(n_back_and_forths: int = 1):
        start = time.perf_counter()
        original = torch.rand(100, 100, 100, device=device)
        v = original.clone()
        for n in range(n_back_and_forths):
            v = torch_to_jax(v)
            v = jax_to_torch(v)
        torch.testing.assert_close(v, original)
        return time.perf_counter() - start

    n_back_and_forths = 100
    time_taken = benchmark(_back_and_forth_loop, n_back_and_forths)
    print(f"Time taken for {n_back_and_forths=} between jax and torch: {time_taken}")
    start = time.perf_counter()
    print(
        f"Time taken for 100 back and forths between jax and torch (with a copy) {time.perf_counter() - start}"
    )
