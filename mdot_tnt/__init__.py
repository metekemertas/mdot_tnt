"""
MDOT-TNT: A Truncated Newton Method for Optimal Transport

This package provides efficient solvers for the entropic-regularized optimal transport
problem, as introduced in the paper "A Truncated Newton Method for Optimal Transport"
by Mete Kemertas, Amir-massoud Farahmand, Allan D. Jepson (ICLR, 2025).
URL: https://openreview.net/forum?id=gWrWUaCbMa

Main functions:
    solve_OT: Solve a single OT problem.
    solve_OT_batched: Solve multiple OT problems simultaneously (5-10x faster).

Example:
    >>> import torch
    >>> from mdot_tnt import solve_OT, solve_OT_batched
    >>>
    >>> # Single problem
    >>> r = torch.rand(512, device='cuda', dtype=torch.float64)
    >>> r = r / r.sum()
    >>> c = torch.rand(512, device='cuda', dtype=torch.float64)
    >>> c = c / c.sum()
    >>> C = torch.rand(512, 512, device='cuda', dtype=torch.float64)
    >>> cost = solve_OT(r, c, C, gamma_f=1024.)
    >>>
    >>> # Batched (32 problems at once)
    >>> r_batch = torch.rand(32, 512, device='cuda', dtype=torch.float64)
    >>> r_batch = r_batch / r_batch.sum(-1, keepdim=True)
    >>> c_batch = torch.rand(32, 512, device='cuda', dtype=torch.float64)
    >>> c_batch = c_batch / c_batch.sum(-1, keepdim=True)
    >>> costs = solve_OT_batched(r_batch, c_batch, C, gamma_f=1024.)
"""

import math
import warnings

import torch as th

from mdot_tnt.batched import solve_OT_batched
from mdot_tnt.mdot import mdot, preprocess_marginals
from mdot_tnt.rounding import round_altschuler, rounded_cost_altschuler

__all__ = ["solve_OT", "solve_OT_batched"]


def solve_OT(r, c, C, gamma_f=1024.0, drop_tiny=False, return_plan=False, round=True, log=False):
    """
    Solve the entropic-regularized optimal transport problem.

    Inputs r, c, C are required to be torch tensors.

    Args:
        r: Row marginal of shape (n,). Must sum to 1.
        c: Column marginal of shape (m,). Must sum to 1.
        C: Cost matrix of shape (n, m). Recommended to scale entries to [0, 1].
        gamma_f: Temperature (inverse of the regularization weight). For many problems,
            stable up to 2^18. Higher values return more accurate solutions but take
            longer to converge. Use double precision if gamma_f is large.
        drop_tiny: If either marginal is known to be sparse, set this to True to drop
            tiny entries for a speedup. If return_plan is True, the returned plan will
            be in the original dimensions.
        return_plan: If True, return the optimal transport plan rather than the cost.
        round: If True, use the rounding algorithm of Altschuler et al. (2017) to
            (a) return a feasible plan if return_plan is True and (b) the cost of
            the rounded plan if return_plan is False.
        log: If True, additionally return a dictionary containing logs of the
            optimization process.

    Returns:
        Transport cost (scalar tensor) if return_plan is False, or the transport
        plan of shape (n, m) if return_plan is True. If log is True, returns a
        tuple of (result, logs_dict).
    """
    assert all(isinstance(x, th.Tensor) for x in [r, c, C]), "r, c, and C must be torch tensors"
    dtype = r.dtype
    # Require high precision for gamma_f > 2^10
    if gamma_f > 2**10 and dtype != th.float64:
        warnings.warn(
            "Switching to double precision for gamma_f > 2^10 during execution. "
            f"Output will be input dtype: {dtype}."
        )
        r, c, C = r.double(), c.double(), C.double()

    if drop_tiny:
        drop_lessthan = math.log(min(r.size(-1), c.size(-1))) / (gamma_f**2)
        (r_, r_keep), (c_, c_keep), C_ = preprocess_marginals(r, c, C, drop_lessthan)

        u_, v_, gamma_f_, k_total, opt_logs = mdot(r_, c_, C_, gamma_f)

        u = -th.ones_like(r) * float("inf")
        u[r_keep] = u_
        v = -th.ones_like(c) * float("inf")
        v[c_keep] = v_
    else:
        u, v, gamma_f_, k_total, opt_logs = mdot(r, c, C, gamma_f)

    # Switch back to original dtype
    u, v = u.to(dtype), v.to(dtype)

    if return_plan:
        P = (u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_f_ * C).exp()
        if round:
            P = round_altschuler(P, r, c)
        if log:
            return P, opt_logs
        return P
    else:
        if round:
            cost = rounded_cost_altschuler(u, v, r, c, C, gamma_f_)
        else:
            cost = ((u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_f_ * C).exp() * C).sum()
        if log:
            return cost, opt_logs
        return cost
