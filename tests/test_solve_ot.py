"""Tests for the single OT solver (solve_OT)."""

import pytest
import torch as th

from mdot_tnt import solve_OT


class TestSolveOT:
    """Test suite for solve_OT function."""

    def test_returns_scalar_cost(self, small_marginals, small_cost_matrix):
        """Test that solve_OT returns a scalar cost by default."""
        r, c = small_marginals
        C = small_cost_matrix
        cost = solve_OT(r, c, C, gamma_f=64.0)
        assert cost.dim() == 0, "Cost should be a scalar"
        assert cost.dtype == r.dtype, "Cost dtype should match input dtype"

    def test_cost_is_nonnegative(self, small_marginals, small_cost_matrix):
        """Test that the returned cost is non-negative."""
        r, c = small_marginals
        C = small_cost_matrix
        cost = solve_OT(r, c, C, gamma_f=64.0)
        assert cost >= 0, "Cost should be non-negative"

    def test_cost_is_finite(self, small_marginals, small_cost_matrix):
        """Test that the returned cost is finite."""
        r, c = small_marginals
        C = small_cost_matrix
        cost = solve_OT(r, c, C, gamma_f=64.0)
        assert th.isfinite(cost), "Cost should be finite"

    def test_return_plan_shape(self, small_marginals, small_cost_matrix):
        """Test that return_plan returns correct shape."""
        r, c = small_marginals
        C = small_cost_matrix
        n, m = r.shape[0], c.shape[0]
        P = solve_OT(r, c, C, gamma_f=64.0, return_plan=True)
        assert P.shape == (n, m), f"Plan should have shape ({n}, {m})"

    def test_plan_is_nonnegative(self, small_marginals, small_cost_matrix):
        """Test that transport plan entries are non-negative."""
        r, c = small_marginals
        C = small_cost_matrix
        P = solve_OT(r, c, C, gamma_f=64.0, return_plan=True)
        assert (P >= -1e-10).all(), "Plan entries should be non-negative"

    def test_rounded_plan_satisfies_marginals(self, small_marginals, small_cost_matrix):
        """Test that rounded plan satisfies marginal constraints."""
        r, c = small_marginals
        C = small_cost_matrix
        P = solve_OT(r, c, C, gamma_f=64.0, return_plan=True, round=True)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        assert th.allclose(row_sums, r, atol=1e-6), "Row sums should match r"
        assert th.allclose(col_sums, c, atol=1e-6), "Column sums should match c"

    def test_log_returns_dict(self, small_marginals, small_cost_matrix):
        """Test that log=True returns optimization logs."""
        r, c = small_marginals
        C = small_cost_matrix
        cost, logs = solve_OT(r, c, C, gamma_f=64.0, log=True)
        assert isinstance(logs, dict), "logs should be a dictionary"
        assert "proj_logs" in logs, "logs should contain proj_logs"

    def test_higher_gamma_lower_cost(self, medium_marginals, medium_cost_matrix):
        """Test that higher gamma_f gives lower (more accurate) cost."""
        r, c = medium_marginals
        C = medium_cost_matrix

        cost_low = solve_OT(r, c, C, gamma_f=64.0)
        cost_high = solve_OT(r, c, C, gamma_f=256.0)

        # Higher gamma should give tighter bound (closer to true OT)
        # The cost should be similar or lower
        assert cost_high <= cost_low + 0.1, "Higher gamma should not increase cost significantly"

    def test_input_validation_requires_tensors(self):
        """Test that non-tensor inputs raise AssertionError."""
        r = [0.5, 0.5]  # List, not tensor
        c = th.tensor([0.5, 0.5])
        C = th.rand(2, 2)
        with pytest.raises(AssertionError):
            solve_OT(r, c, C)

    def test_unrounded_plan_approximately_satisfies_marginals(
        self, small_marginals, small_cost_matrix
    ):
        """Test that unrounded plan approximately satisfies marginals."""
        r, c = small_marginals
        C = small_cost_matrix
        P = solve_OT(r, c, C, gamma_f=256.0, return_plan=True, round=False)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        # Unrounded should be close but not exact
        assert th.allclose(row_sums, r, atol=1e-3), "Row sums should approximately match r"
        assert th.allclose(col_sums, c, atol=1e-3), "Column sums should approximately match c"

    def test_deterministic_results(self, small_marginals, small_cost_matrix):
        """Test that results are deterministic given same inputs."""
        r, c = small_marginals
        C = small_cost_matrix

        cost1 = solve_OT(r, c, C, gamma_f=64.0)
        cost2 = solve_OT(r, c, C, gamma_f=64.0)

        assert th.allclose(cost1, cost2), "Results should be deterministic"

    def test_high_precision_regime(self, medium_marginals, medium_cost_matrix):
        """Test solver in high precision regime (gamma_f=1024)."""
        r, c = medium_marginals
        C = medium_cost_matrix

        cost = solve_OT(r, c, C, gamma_f=1024.0)

        assert th.isfinite(cost), "Cost should be finite in high precision regime"
        assert cost >= 0, "Cost should be non-negative"

    def test_high_precision_plan_satisfies_marginals(self, medium_marginals, medium_cost_matrix):
        """Test that high precision plan satisfies marginal constraints tightly."""
        r, c = medium_marginals
        C = medium_cost_matrix

        P = solve_OT(r, c, C, gamma_f=1024.0, return_plan=True, round=True)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        assert th.allclose(row_sums, r, atol=1e-6), "Row sums should match r"
        assert th.allclose(col_sums, c, atol=1e-6), "Column sums should match c"
        assert (P >= -1e-10).all(), "Plan entries should be non-negative"
