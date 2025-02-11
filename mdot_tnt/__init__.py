"""
Code for solving the entropic-regularized optimal transport problem via the MDOT-TruncatedNewton (MDOT-TNT)
method introduced in the paper "A Truncated Newton Method for Optimal Transport"
by Mete Kemertas, Amir-massoud Farahmand, Allan D. Jepson (ICLR, 2025).
URL: https://openreview.net/forum?id=gWrWUaCbMa
"""

import math
import torch as th
import warnings

from mdot_tnt.mdot import mdot, preprocess_marginals
from mdot_tnt.rounding import round_altschuler, rounded_cost_altschuler


def solve_OT(r, c, C, gamma_f=1024., drop_tiny=False, return_plan=False, round=True, log=False):
    """
    Solve the entropic-regularized optimal transport problem. Inputs r, c, C are required to be torch tensors.
    :param r: n-dimensional row marginal.
    :param c: m-dimensional column marginal.
    :param C: n x m cost matrix. Recommended use is to scale the entries to be in [0, 1].
    :param gamma_f: The temperature (inverse of the regularization weight). For many problems, stable up to 2^18.
    Higher values return more accurate solutions but take longer to converge. Use double precision if gamma_f large.
    :param drop_tiny: If either marginal is known to be sparse, set this to True to drop tiny entries for a speedup.
    If return_plan is True, the returned plan will be in the original dimensions.
    :param return_plan: If True, return the optimal transport plan rather than the cost.
    :param round: If True, use the rounding algorithm of Altschuler et al. (2017) to (a) return a feasible plan
    if return_plan is True and (b) the cost of the rounded plan if return_plan is False.
    :param log: If True, additionally return a dictionary containing logs of the optimization process.
    :return: If return_plan is True, return the optimal transport plan as a torch tensor. Otherwise, return the cost.
    """
    assert all(isinstance(x, th.Tensor) for x in [r, c, C]), "r, c, and C must be torch tensors"
    dtype = r.dtype
    # Require high precision for gamma_f > 2^10
    if gamma_f > 2 ** 10 and dtype != th.float64:
        warnings.warn("Switching to double precision for gamma_f > 2^10 during execution. "
                      "Output will be input dtype: {}.".format(dtype))
        r, c, C = r.double(), c.double(), C.double()

    if drop_tiny:
        drop_lessthan = math.log(min(r.size(-1), c.size(-1))) / (gamma_f ** 2)
        (r_, r_keep), (c_, c_keep), C_ = preprocess_marginals(r, c, C, drop_lessthan)

        u_, v_, gamma_f_, k_total, opt_logs = mdot(r_, c_, C_, gamma_f)

        u = -th.ones_like(r) * float('inf')
        u[r_keep] = u_
        v = -th.ones_like(c) * float('inf')
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
