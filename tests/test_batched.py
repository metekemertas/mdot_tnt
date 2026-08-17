"""Tests for the batched OT solver (solve_OT_batched)."""

import pytest
import torch as th

from mdot_tnt import solve_OT, solve_OT_batched


class TestSolveOTBatched:
    """Test suite for solve_OT_batched function."""

    def test_returns_batch_costs(self, batched_marginals, batched_cost_matrix):
        """Test that solve_OT_batched returns costs for each problem."""
        r, c = batched_marginals
        C = batched_cost_matrix
        batch_size = r.shape[0]

        costs = solve_OT_batched(r, c, C, gamma_f=64.0)

        assert costs.shape == (batch_size,), f"Should return {batch_size} costs"
        assert costs.dtype == r.dtype, "Cost dtype should match input dtype"

    def test_costs_are_nonnegative(self, batched_marginals, batched_cost_matrix):
        """Test that all returned costs are non-negative."""
        r, c = batched_marginals
        C = batched_cost_matrix
        costs = solve_OT_batched(r, c, C, gamma_f=64.0)
        assert (costs >= 0).all(), "All costs should be non-negative"

    def test_costs_are_finite(self, batched_marginals, batched_cost_matrix):
        """Test that all returned costs are finite."""
        r, c = batched_marginals
        C = batched_cost_matrix
        costs = solve_OT_batched(r, c, C, gamma_f=64.0)
        assert th.isfinite(costs).all(), "All costs should be finite"

    def test_return_plan_shape(self, batched_marginals, batched_cost_matrix):
        """Test that return_plan returns correct shape."""
        r, c = batched_marginals
        C = batched_cost_matrix
        batch_size, n = r.shape
        m = c.shape[1]

        P = solve_OT_batched(r, c, C, gamma_f=64.0, return_plan=True)

        assert P.shape == (batch_size, n, m), f"Plan should have shape ({batch_size}, {n}, {m})"

    def test_plan_is_nonnegative(self, batched_marginals, batched_cost_matrix):
        """Test that all transport plan entries are non-negative."""
        r, c = batched_marginals
        C = batched_cost_matrix
        P = solve_OT_batched(r, c, C, gamma_f=64.0, return_plan=True)
        assert (P >= -1e-10).all(), "Plan entries should be non-negative"

    def test_rounded_plan_satisfies_marginals(self, batched_marginals, batched_cost_matrix):
        """Test that rounded plans satisfy marginal constraints."""
        r, c = batched_marginals
        C = batched_cost_matrix
        P = solve_OT_batched(r, c, C, gamma_f=64.0, return_plan=True, round=True)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        assert th.allclose(row_sums, r, atol=1e-5), "Row sums should match r"
        assert th.allclose(col_sums, c, atol=1e-5), "Column sums should match c"

    def test_log_returns_dict(self, batched_marginals, batched_cost_matrix):
        """Test that log=True returns optimization logs."""
        r, c = batched_marginals
        C = batched_cost_matrix
        costs, logs = solve_OT_batched(r, c, C, gamma_f=64.0, log=True)
        assert isinstance(logs, dict), "logs should be a dictionary"

    def test_consistency_with_single_solver(self, device, dtype):
        """Test that batched solver gives similar results to single solver."""
        n = 16
        r = th.rand(n, device=device, dtype=dtype)
        r = r / r.sum()
        c = th.rand(n, device=device, dtype=dtype)
        c = c / c.sum()
        C = th.rand(n, n, device=device, dtype=dtype)
        C = C / C.max()

        # Single solver
        cost_single = solve_OT(r, c, C, gamma_f=64.0)

        # Batched solver with batch_size=1
        r_batch = r.unsqueeze(0)
        c_batch = c.unsqueeze(0)
        costs_batched = solve_OT_batched(r_batch, c_batch, C, gamma_f=64.0)

        assert th.allclose(cost_single, costs_batched[0], rtol=1e-2), (
            "Batched solver should match single solver"
        )

    def test_per_problem_cost_matrix(self, batched_marginals, device, dtype):
        """Test with per-problem cost matrices (batch, n, m)."""
        r, c = batched_marginals
        batch_size, n = r.shape
        m = c.shape[1]

        # Create per-problem cost matrices
        C = th.rand(batch_size, n, m, device=device, dtype=dtype)
        C = C / C.amax(dim=(-2, -1), keepdim=True)

        costs = solve_OT_batched(r, c, C, gamma_f=64.0)

        assert costs.shape == (batch_size,), "Should return one cost per problem"
        assert (costs >= 0).all(), "All costs should be non-negative"
        assert th.isfinite(costs).all(), "All costs should be finite"

    def test_input_validation_r_shape(self, device, dtype):
        """Test that 1D r raises ValueError."""
        r = th.rand(10, device=device, dtype=dtype)  # 1D, should be 2D
        c = th.rand(1, 10, device=device, dtype=dtype)
        C = th.rand(10, 10, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="r must be 2D"):
            solve_OT_batched(r, c, C)

    def test_input_validation_c_shape(self, device, dtype):
        """Test that 1D c raises ValueError."""
        r = th.rand(1, 10, device=device, dtype=dtype)
        c = th.rand(10, device=device, dtype=dtype)  # 1D, should be 2D
        C = th.rand(10, 10, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="c must be 2D"):
            solve_OT_batched(r, c, C)

    def test_input_validation_batch_mismatch(self, device, dtype):
        """Test that mismatched batch sizes raise ValueError."""
        r = th.rand(5, 10, device=device, dtype=dtype)
        c = th.rand(3, 10, device=device, dtype=dtype)  # Different batch size
        C = th.rand(10, 10, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            solve_OT_batched(r, c, C)

    def test_deterministic_results(self, batched_marginals, batched_cost_matrix):
        """Test that results are deterministic given same inputs."""
        r, c = batched_marginals
        C = batched_cost_matrix

        costs1 = solve_OT_batched(r, c, C, gamma_f=64.0)
        costs2 = solve_OT_batched(r, c, C, gamma_f=64.0)

        assert th.allclose(costs1, costs2), "Results should be deterministic"

    def test_high_precision_regime(self, batched_marginals, batched_cost_matrix):
        """Test batched solver in high precision regime (gamma_f=1024)."""
        r, c = batched_marginals
        C = batched_cost_matrix

        costs = solve_OT_batched(r, c, C, gamma_f=1024.0)

        assert th.isfinite(costs).all(), "All costs should be finite in high precision regime"
        assert (costs >= 0).all(), "All costs should be non-negative"

    def test_high_precision_plan_satisfies_marginals(self, batched_marginals, batched_cost_matrix):
        """Test that high precision plans satisfy marginal constraints tightly."""
        r, c = batched_marginals
        C = batched_cost_matrix

        P = solve_OT_batched(r, c, C, gamma_f=1024.0, return_plan=True, round=True)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        assert th.allclose(row_sums, r, atol=1e-5), "Row sums should match r"
        assert th.allclose(col_sums, c, atol=1e-5), "Column sums should match c"
        assert (P >= -1e-10).all(), "Plan entries should be non-negative"
