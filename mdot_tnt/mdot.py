"""Core MDOT solver using truncated Newton projection."""

import warnings
from typing import Any, Dict, List, Tuple, Union

import torch as th

from mdot_tnt.truncated_newton import TruncatedNewtonProjector


def preprocess_marginals(
    r: th.Tensor, c: th.Tensor, C: th.Tensor, eps: float
) -> Tuple[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor], th.Tensor]:
    """
    Drop the smallest marginal entries whose cumulative sum is below a threshold.

    Args:
        r: The row marginal of shape (n,).
        c: The column marginal of shape (m,).
        C: The cost matrix of shape (n, m).
        eps: The threshold for the cumulative sum of the marginal entries to be dropped.

    Returns:
        A tuple containing:
            - (r_new, r_keep): The new row marginal and indices of kept entries.
            - (c_new, c_keep): The new column marginal and indices of kept entries.
            - C: The cost matrix with corresponding rows and columns dropped.
    """

    def preprocess_marginal(m: th.Tensor, eps: float) -> Tuple[th.Tensor, th.Tensor]:
        m_sorted, m_idx = th.sort(m, dim=-1, descending=False)
        m_cumsum = th.cumsum(m_sorted, dim=-1)
        m_keep = m_idx[m_cumsum > eps]
        m_new = m[m_keep]
        mass_removed = 1 - m_new.sum(-1)
        m_new = m_new + mass_removed / m_new.size(-1)

        return m_new, m_keep

    r_new, r_keep = preprocess_marginal(r, eps)
    c_new, c_keep = preprocess_marginal(c, eps)
    print(
        f"Dropped {r.size(-1) - r_new.size(-1)} entries from r and {c.size(-1) - c_new.size(-1)} entries from c."
    )

    C = C[r_keep][:, c_keep]

    return (r_new, r_keep), (c_new, c_keep), C


def smooth_marginals(
    r: th.Tensor,
    c: th.Tensor,
    eps: th.Tensor,
    w_r: float = 0.5,
    w_c: float = 0.5,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Smooth the marginals by adding a small amount of uniform mass to each entry.

    Args:
        r: The row marginal of shape (n,).
        c: The column marginal of shape (m,).
        eps: The amount of mass to add to each entry.
        w_r: The weight for the row marginal.
        w_c: The weight for the column marginal.

    Returns:
        A tuple (r_hat, c_hat) of smoothed marginals with total TV distance at most eps
        from the original marginals.
    """
    assert w_r + w_c == 1, "w_r and w_c must sum to 1"
    eps = eps.clamp(max=1.0).unsqueeze(-1)
    r_hat = (1 - w_r * eps) * r + w_r * eps * th.ones_like(r) / r.size(-1)
    c_hat = (1 - w_c * eps) * c + w_c * eps * th.ones_like(c) / c.size(-1)

    return r_hat, c_hat


def adjust_schedule(q: float, deltas: Union[List[float], None] = None) -> float:
    """
    Adjust the temperature annealing schedule based on the success of the Truncated Newton method.

    Args:
        q: The current temperature annealing schedule adjustment factor.
        deltas: The list of deltas from the Truncated Newton method;
                see Sec. 3.3 of Kemertas et al. (2025).

    Returns:
        The new temperature annealing schedule adjustment factor.
    """
    if deltas is None:
        return q

    deltas = deltas + [1.0]  # If deltas is empty, we assume that the first iteration was successful
    delta_min = min(deltas)

    if delta_min < 0.5:
        q = q**0.5
    elif delta_min > 0.9:
        q = q**2

    return q


def mdot(
    r: th.Tensor,
    c: th.Tensor,
    C: th.Tensor,
    gamma_f: float,
    gamma_i: float = 16,
    p: float = 1.5,
    q: float = 2.0,
) -> Tuple[th.Tensor, th.Tensor, float, int, Dict[str, Any]]:
    """
    Solve the entropic-regularized optimal transport problem using the MDOT method.

    This implements the MDOT method introduced in the paper:
    "Efficient and Accurate Optimal Transport with Mirror Descent and Conjugate Gradients"
    by Mete Kemertas, Allan D. Jepson and Amir-massoud Farahmand.
    URL: https://arxiv.org/abs/2307.08507

    Here, we use the Truncated Newton method for projection.

    Args:
        r: The first marginal of shape (n,).
        c: The second marginal of shape (m,).
        C: The cost matrix of shape (n, m). Recommended to scale entries to [0, 1].
        gamma_f: The final temperature (inverse of the regularization weight).
        gamma_i: The initial temperature.
        p: The exponent for the epsilon function, used to determine the stopping
           criterion for the dual gradient.
        q: The temperature annealing (or mirror descent step size) schedule adjustment factor.

    Returns:
        A tuple containing:
            - u: The row dual variables of shape (n,).
            - v: The column dual variables of shape (m,).
            - gamma: The final temperature achieved.
            - k_total: The total number of O(n^2) primitive operations.
            - logs: Dictionary with optimization statistics.
    """
    projector = TruncatedNewtonProjector(device=C.device, dtype=C.dtype)

    H_r = -(r * (r + 1e-30).log()).sum(-1)
    H_c = -(c * (c + 1e-30).log()).sum(-1)
    H_min = th.min(H_r, H_c)
    eps_fn = lambda g_: H_min / (g_**p)

    logs = {
        "proj_logs": [],
        "eps": [],
    }

    t = 1
    done = False
    gamma = min(gamma_i, gamma_f)
    gammas = [0.0, gamma]

    while not done:
        done = abs(gamma - gamma_f) < 1e-5  # Check if gamma == gamma_f (modulo rounding errors)

        eps_d = eps_fn(gamma)

        r_hat, c_hat = smooth_marginals(r, c, eps_d / 2, w_r=0.9, w_c=0.1)

        if t == 1:
            u_init, v_init = r_hat.log(), c_hat.log()
            u_cur, v_cur = u_init.clone(), v_init.clone()

        u_prev, v_prev = u_cur.clone(), v_cur.clone()
        gamma_C = gamma * C
        u_cur, v_cur, proj_log, success = projector.project(
            gamma_C, r_hat.log(), c_hat.log(), eps_d / 2, u_init, v_init
        )

        logs["proj_logs"].append(proj_log)
        if not success:
            warnings.warn(
                f"Projection failed. Returning result at the last temperature: {1 / gammas[-2]:.4e}"
            )
            u_cur = u_prev.clone()
            v_cur = v_prev.clone()
            gammas = gammas[:-1]
            break

        q = adjust_schedule(q, proj_log["deltas"])
        gamma = min(gamma * q, gamma_f)

        if not done:
            # Generate warm-started initializations for the next iteration.
            u_init = u_cur + (u_cur - u_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])
            v_init = v_cur + (v_cur - v_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])

        gammas.append(gamma)
        t += 1

    k_total = sum([log["n_iter"] for log in logs["proj_logs"]])
    k_total += t - 1
    logs["success"] = success
    logs["gammas"] = gammas

    return u_cur, v_cur, gammas[-1], k_total, logs
