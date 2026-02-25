"""Tests for the low-memory OT solver (solve_OT_lowmem)."""

import pytest
import torch as th

from mdot_tnt import solve_OT
from mdot_tnt.lowmem import (
    euclidean,
    solve_OT_lowmem,
    squared_euclidean,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_point_clouds(device, dtype):
    """Source (n=10, d=3) and target (m=12, d=3) point clouds."""
    X = th.rand(10, 3, device=device, dtype=dtype)
    Y = th.rand(12, 3, device=device, dtype=dtype)
    return X, Y


@pytest.fixture
def medium_point_clouds(device, dtype):
    """Source (n=64, d=5) and target (m=64, d=5) point clouds."""
    X = th.rand(64, 5, device=device, dtype=dtype)
    Y = th.rand(64, 5, device=device, dtype=dtype)
    return X, Y


# ===================================================================
# Dense-C mode (backward-compatible with solve_OT)
# ===================================================================


class TestLowmemDenseC:
    """Tests using a pre-computed dense cost matrix."""

    def test_returns_scalar_cost(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        C = small_cost_matrix
        cost = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=4)
        assert cost.dim() == 0, "Cost should be a scalar"
        assert cost.dtype == r.dtype, "Cost dtype should match input"

    def test_cost_is_nonnegative(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        cost = solve_OT_lowmem(r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4)
        assert cost >= 0, "Cost should be non-negative"

    def test_cost_is_finite(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        cost = solve_OT_lowmem(r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4)
        assert th.isfinite(cost), "Cost should be finite"

    def test_return_plan_shape(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        n, m = r.shape[0], c.shape[0]
        P = solve_OT_lowmem(
            r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4, return_plan=True
        )
        assert P.shape == (n, m)

    def test_plan_is_nonnegative(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        P = solve_OT_lowmem(
            r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4, return_plan=True
        )
        assert (P >= -1e-10).all()

    def test_rounded_plan_satisfies_marginals(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        P = solve_OT_lowmem(
            r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4,
            return_plan=True, round=True,
        )
        assert th.allclose(P.sum(-1), r, atol=1e-6)
        assert th.allclose(P.sum(-2), c, atol=1e-6)

    def test_log_returns_dict(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        cost, logs = solve_OT_lowmem(
            r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4, log=True
        )
        assert isinstance(logs, dict)
        assert "proj_logs" in logs

    def test_unrounded_cost(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        cost = solve_OT_lowmem(
            r, c, C=small_cost_matrix, gamma_f=64.0, block_size=4, round=False
        )
        assert th.isfinite(cost) and cost >= 0

    def test_deterministic_results(self, small_marginals, small_cost_matrix):
        r, c = small_marginals
        C = small_cost_matrix
        c1 = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=4)
        c2 = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=4)
        assert th.allclose(c1, c2)


# ===================================================================
# Point-cloud mode
# ===================================================================


class TestLowmemPointCloud:
    """Tests using point clouds (X, Y) — no dense C stored."""

    def _normalized_sq_euc(self, X, Y):
        """Return a normalized cost_fn (values in [0, 1]) for a given (X, Y)."""
        C_full = squared_euclidean(X, Y)
        C_max = C_full.max()
        return lambda X_, Yb_: squared_euclidean(X_, Yb_) / C_max

    def test_returns_scalar_cost(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        cf = self._normalized_sq_euc(X, Y)
        cost = solve_OT_lowmem(r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4)
        assert cost.dim() == 0

    def test_cost_is_nonnegative(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        cf = self._normalized_sq_euc(X, Y)
        cost = solve_OT_lowmem(r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4)
        assert cost >= 0

    def test_cost_is_finite(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        cf = self._normalized_sq_euc(X, Y)
        cost = solve_OT_lowmem(r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4)
        assert th.isfinite(cost)

    def test_return_plan_shape(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        n, m = X.shape[0], Y.shape[0]
        cf = self._normalized_sq_euc(X, Y)
        P = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4, return_plan=True
        )
        assert P.shape == (n, m)

    def test_rounded_plan_satisfies_marginals(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        cf = self._normalized_sq_euc(X, Y)
        P = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4,
            return_plan=True, round=True,
        )
        assert th.allclose(P.sum(-1), r, atol=1e-6)
        assert th.allclose(P.sum(-2), c, atol=1e-6)

    def test_euclidean_cost_fn(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        C_full = euclidean(X, Y)
        C_max = C_full.max()
        cf = lambda X_, Yb_: euclidean(X_, Yb_) / C_max
        cost = solve_OT_lowmem(r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4)
        assert th.isfinite(cost) and cost >= 0

    def test_default_cost_fn_is_squared_euclidean(self, device, dtype):
        """When cost_fn is None, squared_euclidean should be used."""
        n, m, d = 8, 8, 2
        X = th.rand(n, d, device=device, dtype=dtype)
        Y = th.rand(m, d, device=device, dtype=dtype)
        r = th.ones(n, device=device, dtype=dtype) / n
        c = th.ones(m, device=device, dtype=dtype) / m
        # Should not raise
        cost = solve_OT_lowmem(r, c, X=X, Y=Y, gamma_f=64.0, block_size=4)
        assert th.isfinite(cost)


# ===================================================================
# Consistency between dense-C and point-cloud modes
# ===================================================================


class TestLowmemConsistency:
    """Verify that dense-C and point-cloud modes produce identical results."""

    def test_cost_matches_dense(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        C = squared_euclidean(X, Y)
        C_max = C.max()
        C_norm = C / C_max
        cf = lambda X_, Yb_: squared_euclidean(X_, Yb_) / C_max

        cost_dense = solve_OT_lowmem(
            r, c, C=C_norm, gamma_f=64.0, block_size=4
        )
        cost_pc = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4
        )
        assert th.allclose(cost_dense, cost_pc, atol=1e-12)

    def test_plan_matches_dense(self, small_marginals, small_point_clouds):
        r, c = small_marginals
        X, Y = small_point_clouds
        C = squared_euclidean(X, Y)
        C_max = C.max()
        C_norm = C / C_max
        cf = lambda X_, Yb_: squared_euclidean(X_, Yb_) / C_max

        P_d = solve_OT_lowmem(
            r, c, C=C_norm, gamma_f=64.0, block_size=4, return_plan=True
        )
        P_pc = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4,
            return_plan=True,
        )
        assert th.allclose(P_d, P_pc, atol=1e-12)

    def test_matches_standard_solver(self, medium_marginals, medium_cost_matrix):
        """Low-memory dense-C mode should match the standard solve_OT."""
        r, c = medium_marginals
        C = medium_cost_matrix
        cost_std = solve_OT(r, c, C, gamma_f=256.0)
        cost_lm = solve_OT_lowmem(r, c, C=C, gamma_f=256.0, block_size=16)
        assert th.allclose(cost_std, cost_lm, atol=1e-10)


# ===================================================================
# Block-size variations
# ===================================================================


class TestLowmemBlockSizes:
    """Verify correctness across different block sizes."""

    @pytest.mark.parametrize("block_size", [1, 3, 6, 12])
    def test_dense_block_sizes(self, small_marginals, small_cost_matrix, block_size):
        r, c = small_marginals
        C = small_cost_matrix
        cost = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=block_size)
        assert th.isfinite(cost) and cost >= 0

    @pytest.mark.parametrize("block_size", [1, 3, 6, 12])
    def test_block_sizes_agree(self, small_marginals, small_cost_matrix, block_size):
        """All block sizes should produce the same cost (up to float rounding)."""
        r, c = small_marginals
        C = small_cost_matrix
        cost_full = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=C.shape[-1])
        cost_blk = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=block_size)
        assert th.allclose(cost_full, cost_blk, atol=1e-10)

    @pytest.mark.parametrize("block_size", [1, 4, 12])
    def test_pointcloud_block_sizes(self, small_marginals, small_point_clouds, block_size):
        r, c = small_marginals
        X, Y = small_point_clouds
        C_max = squared_euclidean(X, Y).max()
        cf = lambda X_, Yb_: squared_euclidean(X_, Yb_) / C_max
        cost = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=block_size
        )
        assert th.isfinite(cost) and cost >= 0


# ===================================================================
# High-precision regime
# ===================================================================


class TestLowmemHighPrecision:
    """Tests at high gamma_f (triggers float64 promotion)."""

    def test_high_gamma_dense(self, medium_marginals, medium_cost_matrix):
        r, c = medium_marginals
        C = medium_cost_matrix
        cost = solve_OT_lowmem(r, c, C=C, gamma_f=65536.0, block_size=16)
        assert th.isfinite(cost) and cost >= 0

    def test_high_gamma_pointcloud(self, medium_marginals, medium_point_clouds):
        r, c = medium_marginals
        X, Y = medium_point_clouds
        C_max = squared_euclidean(X, Y).max()
        cf = lambda X_, Yb_: squared_euclidean(X_, Yb_) / C_max
        cost = solve_OT_lowmem(
            r, c, X=X, Y=Y, cost_fn=cf, gamma_f=65536.0, block_size=16
        )
        assert th.isfinite(cost) and cost >= 0

    def test_high_gamma_plan_satisfies_marginals(
        self, medium_marginals, medium_cost_matrix
    ):
        r, c = medium_marginals
        C = medium_cost_matrix
        P = solve_OT_lowmem(
            r, c, C=C, gamma_f=65536.0, block_size=16,
            return_plan=True, round=True,
        )
        assert th.allclose(P.sum(-1), r, atol=1e-6)
        assert th.allclose(P.sum(-2), c, atol=1e-6)
        assert (P >= -1e-10).all()

    def test_higher_gamma_lower_cost(self, medium_marginals, medium_cost_matrix):
        r, c = medium_marginals
        C = medium_cost_matrix
        cost_low = solve_OT_lowmem(r, c, C=C, gamma_f=64.0, block_size=16)
        cost_high = solve_OT_lowmem(r, c, C=C, gamma_f=256.0, block_size=16)
        assert cost_high <= cost_low + 0.1


# ===================================================================
# Cost functions
# ===================================================================


class TestCostFunctions:
    """Tests for the built-in cost function implementations."""

    def test_squared_euclidean_shape(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = squared_euclidean(X, Y)
        assert C.shape == (10, 7)

    def test_squared_euclidean_nonnegative(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = squared_euclidean(X, Y)
        assert (C >= -1e-12).all()

    def test_squared_euclidean_zero_diagonal(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        C = squared_euclidean(X, X)
        assert th.allclose(C.diag(), th.zeros(10, device=device, dtype=dtype), atol=1e-12)

    def test_squared_euclidean_matches_cdist(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = squared_euclidean(X, Y)
        C_ref = th.cdist(X, Y, p=2.0) ** 2
        assert th.allclose(C, C_ref, atol=1e-10)

    def test_euclidean_shape(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = euclidean(X, Y)
        assert C.shape == (10, 7)

    def test_euclidean_nonnegative(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = euclidean(X, Y)
        assert (C >= -1e-12).all()

    def test_euclidean_matches_cdist(self, device, dtype):
        X = th.rand(10, 5, device=device, dtype=dtype)
        Y = th.rand(7, 5, device=device, dtype=dtype)
        C = euclidean(X, Y)
        C_ref = th.cdist(X, Y, p=2.0)
        assert th.allclose(C, C_ref, atol=1e-10)

    def test_custom_cost_fn(self, small_marginals, small_point_clouds):
        """Solver accepts an arbitrary user-defined cost function."""
        r, c = small_marginals
        X, Y = small_point_clouds
        # L1 (Manhattan) distance
        def l1_cost(X_, Yb_):
            return th.cdist(X_.unsqueeze(0), Yb_.unsqueeze(0), p=1.0).squeeze(0)
        C_max = l1_cost(X, Y).max()
        cf = lambda X_, Yb_: l1_cost(X_, Yb_) / C_max

        cost = solve_OT_lowmem(r, c, X=X, Y=Y, cost_fn=cf, gamma_f=64.0, block_size=4)
        assert th.isfinite(cost) and cost >= 0


# ===================================================================
# Input validation
# ===================================================================


class TestLowmemValidation:
    """Input validation and error handling."""

    def test_rejects_both_C_and_XY(self, device, dtype):
        n = 8
        r = th.ones(n, device=device, dtype=dtype) / n
        c = th.ones(n, device=device, dtype=dtype) / n
        C = th.rand(n, n, device=device, dtype=dtype)
        X = th.rand(n, 3, device=device, dtype=dtype)
        Y = th.rand(n, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Provide exactly one"):
            solve_OT_lowmem(r, c, C=C, X=X, Y=Y)

    def test_rejects_no_cost_input(self, device, dtype):
        n = 8
        r = th.ones(n, device=device, dtype=dtype) / n
        c = th.ones(n, device=device, dtype=dtype) / n
        with pytest.raises(ValueError, match="Provide exactly one"):
            solve_OT_lowmem(r, c)

    def test_rejects_X_without_Y(self, device, dtype):
        n = 8
        r = th.ones(n, device=device, dtype=dtype) / n
        c = th.ones(n, device=device, dtype=dtype) / n
        X = th.rand(n, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Provide exactly one"):
            solve_OT_lowmem(r, c, X=X)

    def test_rejects_Y_without_X(self, device, dtype):
        n = 8
        r = th.ones(n, device=device, dtype=dtype) / n
        c = th.ones(n, device=device, dtype=dtype) / n
        Y = th.rand(n, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Provide exactly one"):
            solve_OT_lowmem(r, c, Y=Y)
