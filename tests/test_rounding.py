"""Tests for the rounding functions."""

import torch as th

from mdot_tnt.rounding import round_altschuler, rounded_cost_altschuler


class TestRoundAltschuler:
    """Test suite for round_altschuler function."""

    def test_output_shape_matches_input(self, small_marginals):
        """Test that output shape matches input shape."""
        r, c = small_marginals
        n, m = r.shape[0], c.shape[0]

        # Create an approximate plan
        P = th.rand(n, m, device=r.device, dtype=r.dtype)
        P = P / P.sum()

        P_rounded = round_altschuler(P.clone(), r, c)

        assert P_rounded.shape == (n, m), "Output shape should match input"

    def test_satisfies_row_marginal(self, small_marginals):
        """Test that rounded plan satisfies row marginal constraint."""
        r, c = small_marginals
        n, m = r.shape[0], c.shape[0]

        P = th.rand(n, m, device=r.device, dtype=r.dtype)
        P = P / P.sum()

        P_rounded = round_altschuler(P.clone(), r, c)
        row_sums = P_rounded.sum(dim=-1)

        assert th.allclose(row_sums, r, atol=1e-6), "Row sums should match r"

    def test_satisfies_column_marginal(self, small_marginals):
        """Test that rounded plan satisfies column marginal constraint."""
        r, c = small_marginals
        n, m = r.shape[0], c.shape[0]

        P = th.rand(n, m, device=r.device, dtype=r.dtype)
        P = P / P.sum()

        P_rounded = round_altschuler(P.clone(), r, c)
        col_sums = P_rounded.sum(dim=-2)

        assert th.allclose(col_sums, c, atol=1e-6), "Column sums should match c"

    def test_entries_are_nonnegative(self, small_marginals):
        """Test that rounded plan has non-negative entries."""
        r, c = small_marginals
        n, m = r.shape[0], c.shape[0]

        P = th.rand(n, m, device=r.device, dtype=r.dtype)
        P = P / P.sum()

        P_rounded = round_altschuler(P.clone(), r, c)

        assert (P_rounded >= -1e-10).all(), "All entries should be non-negative"

    def test_preserves_dtype(self, device, dtype):
        """Test that rounding preserves dtype."""
        n, m = 5, 7
        r = th.rand(n, device=device, dtype=dtype)
        r = r / r.sum()
        c = th.rand(m, device=device, dtype=dtype)
        c = c / c.sum()
        P = th.rand(n, m, device=device, dtype=dtype)
        P = P / P.sum()

        P_rounded = round_altschuler(P.clone(), r, c)

        assert P_rounded.dtype == dtype, "Dtype should be preserved"


class TestRoundedCostAltschuler:
    """Test suite for rounded_cost_altschuler function."""

    def test_returns_scalar(self, small_marginals, small_cost_matrix):
        """Test that rounded_cost_altschuler returns a scalar."""
        r, c = small_marginals
        C = small_cost_matrix
        n, m = r.shape[0], c.shape[0]

        # Create dual variables
        u = th.randn(n, device=r.device, dtype=r.dtype)
        v = th.randn(m, device=r.device, dtype=r.dtype)
        gamma = 64.0

        cost = rounded_cost_altschuler(u.clone(), v.clone(), r, c, C, gamma)

        assert cost.dim() == 0, "Cost should be a scalar"

    def test_cost_is_nonnegative(self, small_marginals, small_cost_matrix):
        """Test that computed cost is non-negative."""
        r, c = small_marginals
        C = small_cost_matrix

        # Initialize dual variables near log-marginals
        u = r.log()
        v = c.log()
        gamma = 64.0

        cost = rounded_cost_altschuler(u.clone(), v.clone(), r, c, C, gamma)

        assert cost >= -1e-6, "Cost should be non-negative"

    def test_cost_is_finite(self, small_marginals, small_cost_matrix):
        """Test that computed cost is finite."""
        r, c = small_marginals
        C = small_cost_matrix

        u = r.log()
        v = c.log()
        gamma = 64.0

        cost = rounded_cost_altschuler(u.clone(), v.clone(), r, c, C, gamma)

        assert th.isfinite(cost), "Cost should be finite"

    def test_cost_upper_bounded_by_max_cost(self, small_marginals, small_cost_matrix):
        """Test that cost is bounded by maximum possible cost."""
        r, c = small_marginals
        C = small_cost_matrix

        u = r.log()
        v = c.log()
        gamma = 64.0

        cost = rounded_cost_altschuler(u.clone(), v.clone(), r, c, C, gamma)

        # Maximum cost is when all mass goes through highest cost entry
        max_possible_cost = C.max()
        assert cost <= max_possible_cost + 1e-3, "Cost should not exceed max cost"

    def test_preserves_dtype(self, device, dtype):
        """Test that cost computation preserves dtype."""
        n, m = 5, 7
        r = th.rand(n, device=device, dtype=dtype)
        r = r / r.sum()
        c = th.rand(m, device=device, dtype=dtype)
        c = c / c.sum()
        C = th.rand(n, m, device=device, dtype=dtype)
        u = r.log()
        v = c.log()
        gamma = 64.0

        cost = rounded_cost_altschuler(u.clone(), v.clone(), r, c, C, gamma)

        assert cost.dtype == dtype, "Dtype should be preserved"
