"""Shared fixtures for mdot_tnt tests."""

import pytest
import torch as th


@pytest.fixture
def device() -> th.device:
    """Return the device to use for tests (GPU if available, else CPU)."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> th.dtype:
    """Return the default dtype for tests."""
    return th.float64


@pytest.fixture
def small_marginals(device: th.device, dtype: th.dtype):
    """Create small test marginals (n=10, m=12)."""
    n, m = 10, 12
    r = th.rand(n, device=device, dtype=dtype)
    r = r / r.sum()
    c = th.rand(m, device=device, dtype=dtype)
    c = c / c.sum()
    return r, c


@pytest.fixture
def small_cost_matrix(device: th.device, dtype: th.dtype):
    """Create a small cost matrix (10x12)."""
    n, m = 10, 12
    C = th.rand(n, m, device=device, dtype=dtype)
    C = C / C.max()  # Normalize to [0, 1]
    return C


@pytest.fixture
def medium_marginals(device: th.device, dtype: th.dtype):
    """Create medium test marginals (n=64, m=64)."""
    n = 64
    r = th.rand(n, device=device, dtype=dtype)
    r = r / r.sum()
    c = th.rand(n, device=device, dtype=dtype)
    c = c / c.sum()
    return r, c


@pytest.fixture
def medium_cost_matrix(device: th.device, dtype: th.dtype):
    """Create a medium cost matrix (64x64)."""
    n = 64
    C = th.rand(n, n, device=device, dtype=dtype)
    C = C / C.max()
    return C


@pytest.fixture
def batched_marginals(device: th.device, dtype: th.dtype):
    """Create batched test marginals (batch=8, n=32, m=32)."""
    batch_size, n, m = 8, 32, 32
    r = th.rand(batch_size, n, device=device, dtype=dtype)
    r = r / r.sum(dim=-1, keepdim=True)
    c = th.rand(batch_size, m, device=device, dtype=dtype)
    c = c / c.sum(dim=-1, keepdim=True)
    return r, c


@pytest.fixture
def batched_cost_matrix(device: th.device, dtype: th.dtype):
    """Create a shared cost matrix for batched problems (32x32)."""
    n = 32
    C = th.rand(n, n, device=device, dtype=dtype)
    C = C / C.max()
    return C
