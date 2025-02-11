"""
This file contains the implementation of the rounding algorithm proposed by Altschuler et al. (2017) in the paper
"Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration". The algorithm is used to
round the transport plan obtained from the Sinkhorn algorithm to a feasible transport plan in the set U(r, c), where r
and c are the row and column marginals, respectively. The algorithm is used in the mdot.py file to round the transport
plan and compute the cost of the rounded plan. The implementation is based on the original paper.
"""

import torch as th


def round_altschuler(P, r, c):
    """
    Performs rounding given a transport plan and marginals.
    :param P: the input transport plan
    :param r: row marginal
    :param c: column marginal
    :return: rounded transport plan in feasible set U(r, c).
    """
    X = th.min(r / P.sum(-1), th.ones_like(r))
    P *= X.unsqueeze(-1)

    Y = th.min(c / P.sum(-2), th.ones_like(c))
    P *= Y.unsqueeze(-2)

    err_r = (r - P.sum(-1)).clamp(min=0)
    err_c = (c - P.sum(-2)).clamp(min=0)
    P += err_r.unsqueeze(-1) @ err_c.unsqueeze(-2) / (err_r.norm(p=1, dim=-1, keepdim=True) + 1e-30).unsqueeze(-1)

    return P


def rounded_cost_altschuler(u, v, r, c, C, gamma):
    """Performs rounding and cost computation in logdomain given dual variables, without storing n^2 matrices.
    :param u: dual variable for rows
    :param v: dual variable for columns
    :param r: row marginal
    :param c: column marginal
    :param C: cost matrix
    :param gamma: temperature, i.e., the inverse of the entropic regularization weight.
    """
    r_P_log = u + th.logsumexp(v.unsqueeze(-2) - gamma * C, dim=-1)
    delta_u = th.min(r.log() - r_P_log, th.zeros_like(r))
    u += delta_u

    c_P_log = v + th.logsumexp(u.unsqueeze(-1) - gamma * C, dim=-2)
    delta_v = th.min(c.log() - c_P_log, th.zeros_like(c))
    v += delta_v

    r_P_log = u + th.logsumexp(v.unsqueeze(-2) - gamma * C, dim=-1)
    r_P = r_P_log.exp()
    err_r = r - r_P
    err_r /= (err_r.norm(p=1, dim=-1, keepdim=True) + 1e-30)

    c_P_log = v + th.logsumexp(u.unsqueeze(-1) - gamma * C, dim=-2)
    c_P = c_P_log.exp()
    err_c = c - c_P

    cost = th.logsumexp(u.unsqueeze(-1) + v.unsqueeze(-2) - gamma * C + C.log(), dim=(-1, -2)).exp()
    cost += (err_r.unsqueeze(-2) @ C @ err_c.unsqueeze(-1)).sum(-1).sum(-1)

    return cost
