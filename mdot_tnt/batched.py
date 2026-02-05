"""
Batched MDOT-TNT solver for solving multiple optimal transport problems simultaneously.

This module provides batched versions of the MDOT-TNT solver that achieve significant
speedups (5-10x) over sequential solving by amortizing GPU synchronization overhead
across all problems in a batch.

Key insight: The main solver has many Python while-loops that check convergence,
each requiring a GPU→CPU sync. By batching N problems together, we do one sync
per iteration for the entire batch instead of N syncs.

Supports:
- Multiple marginal pairs with shared cost matrix: r, c shape (batch, n), C shape (n, m)
- Multiple OT problems with different costs: r, c shape (batch, n), C shape (batch, n, m)

Example usage:
    >>> import torch
    >>> from mdot_tnt.batched import solve_OT_batched
    >>>
    >>> # 32 problems, each 512-dimensional
    >>> r = torch.rand(32, 512, device='cuda', dtype=torch.float64)
    >>> r = r / r.sum(dim=-1, keepdim=True)
    >>> c = torch.rand(32, 512, device='cuda', dtype=torch.float64)
    >>> c = c / c.sum(dim=-1, keepdim=True)
    >>> C = torch.rand(512, 512, device='cuda', dtype=torch.float64)  # Shared cost
    >>>
    >>> costs = solve_OT_batched(r, c, C, gamma_f=1024.)
    >>> print(costs.shape)  # (32,)
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch as th


class BatchedTruncatedNewtonProjector:
    """
    Batched Truncated Newton projector for the MDOT algorithm.

    Projects onto the set of couplings satisfying marginal constraints,
    processing multiple problems simultaneously for efficiency.
    """

    def __init__(self, device: th.device, dtype: th.dtype, **kwargs):
        """
        Initialize the projector.

        Args:
            device: PyTorch device for computations.
            dtype: Data type for tensors.
            **kwargs: Additional options (debug: bool for verbose output).
        """
        self.device = device
        self.dtype = dtype
        self.debug = kwargs.get("debug", False)

    def project(
        self,
        gamma_C: th.Tensor,
        log_r: th.Tensor,
        log_c: th.Tensor,
        eps_d: Union[float, th.Tensor],
        u: th.Tensor,
        v: th.Tensor,
        active_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Any], th.Tensor]:
        """
        Project onto the constraint set for all problems in the batch.

        Args:
            gamma_C: (batch, n, m) or (n, m) cost matrix scaled by gamma.
            log_r: (batch, n) log of row marginals.
            log_c: (batch, m) log of column marginals.
            eps_d: Convergence tolerance, scalar or (batch,) tensor.
            u: (batch, n) initial row dual variables.
            v: (batch, m) initial column dual variables.
            active_mask: (batch,) bool tensor, True for problems to process.

        Returns:
            u: (batch, n) updated row dual variables.
            v: (batch, m) updated column dual variables.
            logs: Dictionary with optimization statistics.
            success: (batch,) bool tensor indicating convergence per problem.
        """
        batch_size = u.shape[0]

        if active_mask is None:
            active_mask = th.ones(batch_size, device=self.device, dtype=th.bool)

        # Normalize eps_d to (batch,) tensor
        eps_d = self._to_batch_tensor(eps_d, batch_size)

        logs = {"n_iter": 0, "errs": [], "deltas": []}

        # Handle shared vs per-problem cost matrix
        if gamma_C.dim() == 2:
            gamma_C = gamma_C.unsqueeze(0)

        r = log_r.exp()
        c = log_c.exp()

        # Define batched LSE operations
        def LSE_r(v_):
            return th.logsumexp(v_.unsqueeze(-2) - gamma_C, dim=-1)

        def LSE_c(u_):
            return th.logsumexp(u_.unsqueeze(-1) - gamma_C, dim=-2)

        # Initial Sinkhorn step to ensure c = c(P)
        log_c_P = v + LSE_c(u)
        v = v + log_c - log_c_P
        log_r_P = u + LSE_r(v)
        k = 8

        # Chi-Sinkhorn initialization phase
        u, v, log_r_P, err = self._chi_sinkhorn_batched(
            u, v, log_r, log_c, log_r_P, eps_d ** (2 / 5), LSE_r, LSE_c, active_mask
        )
        r_P = log_r_P.exp()
        logs["errs"].append(err.max().item())
        k += 8 * 10

        converged = err <= eps_d
        success = converged.clone()

        num_iter = 0
        max_iter = 100

        # Main Newton loop
        while (~converged & active_mask).any() and num_iter < max_iter:
            num_iter += 1
            working = ~converged & active_mask

            eta_k = th.clamp(err, min=0.9 * eps_d / err.clamp(min=1e-30))
            grad_k = r_P - r

            # Compute transport plan for Hessian
            P = th.exp(u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_C)
            diag_PPc = ((P**2) / c.unsqueeze(-2)).sum(-1)
            k += 8

            # Newton solve
            delta_u, delta_v, matmul_cnt, pcg_success = self._newton_solve_batched(
                P, c, diag_PPc, grad_k, r_P, err, eta_k, working
            )
            success = success & (pcg_success | ~working)
            k += matmul_cnt

            # Line search with Armijo condition
            alpha = th.ones(batch_size, device=self.device, dtype=self.dtype)
            log_c_P = v + alpha.unsqueeze(-1) * delta_v + LSE_c(u + alpha.unsqueeze(-1) * delta_u)
            k += 4

            linear_decr = -(grad_k * delta_u).sum(-1)
            armijo = (log_c_P.exp().sum(-1) - 1) <= (0.99 * alpha * linear_decr)
            armijo = armijo | ~working

            ls_iter = 0
            while not armijo.all() and ls_iter < 20:
                alpha = th.where(armijo, alpha, alpha * 0.5)
                log_c_P = (
                    v + alpha.unsqueeze(-1) * delta_v + LSE_c(u + alpha.unsqueeze(-1) * delta_u)
                )
                k += 4
                armijo = (log_c_P.exp().sum(-1) - 1) <= (0.99 * alpha * linear_decr)
                armijo = armijo | ~working
                ls_iter += 1

            # Update dual variables for working problems
            u = th.where(working.unsqueeze(-1), u + alpha.unsqueeze(-1) * delta_u, u)
            v = th.where(working.unsqueeze(-1), v + alpha.unsqueeze(-1) * delta_v, v)

            # Sinkhorn correction
            v = th.where(working.unsqueeze(-1), v + log_c - log_c_P, v)

            log_r_P = u + LSE_r(v)
            k += 4

            # Chi-Sinkhorn refinement
            u, v, log_r_P, err = self._chi_sinkhorn_batched(
                u, v, log_r, log_c, log_r_P, eps_d ** (2 / 5), LSE_r, LSE_c, working
            )
            r_P = log_r_P.exp()
            logs["errs"].append(err.max().item())

            converged = converged | (err <= eps_d)

        logs["n_iter"] = k

        # Final row update
        delta_u = log_r - log_r_P
        u = u + delta_u

        success = success | converged
        return u, v, logs, success

    def _to_batch_tensor(self, val: Union[float, th.Tensor], batch_size: int) -> th.Tensor:
        """Convert scalar or tensor to (batch,) shaped tensor."""
        if not isinstance(val, th.Tensor):
            val = th.tensor(val, device=self.device, dtype=self.dtype)
        if val.dim() == 0:
            val = val.expand(batch_size)
        return val

    def _chi_sinkhorn_batched(
        self, u, v, log_r, log_c, log_r_P, eps_chi, LSE_r, LSE_c, active_mask, max_iter=100
    ):
        """Batched chi-squared Sinkhorn iterations for initialization."""
        r = log_r.exp()
        r_P = log_r_P.exp()

        err = (r - r_P).abs().sum(-1)
        chi_squared = ((r - r_P) ** 2 / r_P.clamp(min=1e-30)).sum(-1)

        eps_chi = self._to_batch_tensor(eps_chi, u.shape[0])
        working = (chi_squared > eps_chi) & active_mask

        for _ in range(max_iter):
            if not working.any():
                break

            delta_u = log_r - log_r_P
            u = th.where(working.unsqueeze(-1), u + delta_u, u)

            log_c_P = v + LSE_c(u)
            delta_v = log_c - log_c_P
            v = th.where(working.unsqueeze(-1), v + delta_v, v)

            log_r_P = u + LSE_r(v)
            r_P = log_r_P.exp()

            err = (r - r_P).abs().sum(-1)
            chi_squared = ((r - r_P) ** 2 / r_P.clamp(min=1e-30)).sum(-1)
            working = (chi_squared > eps_chi) & active_mask

        return u, v, log_r_P, err

    def _newton_solve_batched(
        self, P, c, diag_PPc, grad_k, r_P, err, eta_k, active_mask, max_iter=50
    ):
        """Batched preconditioned conjugate gradient Newton solve."""
        tol = err * eta_k

        # Diagonal preconditioner
        M_rho = r_P - diag_PPc
        M_rho = th.where(M_rho > 0, M_rho, M_rho.clamp(min=1e-10))

        x = -grad_k / M_rho
        r_vec = r_P * x - self._batched_PPc_matmul(P, c, x) + grad_k
        matmul_cnt = 2

        y = r_vec / M_rho
        p = -y.clone()
        ry_old = (r_vec * y).sum(-1, keepdim=True)

        for _ in range(max_iter):
            PPc_p = self._batched_PPc_matmul(P, c, p)
            matmul_cnt += 2

            Fr_p = r_P * p - PPc_p
            quad = (Fr_p * p).sum(-1, keepdim=True)
            quad = th.where(quad > 0, quad, th.ones_like(quad))

            alpha = ry_old / quad
            x = x + alpha * p
            r_vec = r_vec + alpha * Fr_p

            r_norm = r_vec.abs().sum(-1)
            if (r_norm <= tol).all():
                break

            y = r_vec / M_rho
            ry_new = (r_vec * y).sum(-1, keepdim=True)
            p = -y + (ry_new / ry_old.clamp(min=1e-30)) * p
            ry_old = ry_new

        Pc_x = (x.unsqueeze(-2) @ P).squeeze(-2) / c

        # Track convergence: success if residual norm is below tolerance
        r_norm = r_vec.abs().sum(-1)
        success = r_norm <= tol

        return x, -Pc_x, matmul_cnt, success

    def _batched_PPc_matmul(self, P, c, x):
        """Compute P @ (P^T @ x / c) efficiently in batched form."""
        PTx = (x.unsqueeze(-1) * P).sum(-2)
        PTx_over_c = PTx / c
        return (PTx_over_c.unsqueeze(-2) * P).sum(-1)


def _batched_smooth_marginals(
    r: th.Tensor, c: th.Tensor, eps: th.Tensor, w_r: float = 0.5, w_c: float = 0.5
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Smooth marginals by mixing with uniform distribution.

    Args:
        r: (batch, n) row marginals.
        c: (batch, m) column marginals.
        eps: (batch,) or scalar smoothing factor.
        w_r, w_c: Weights for row/column smoothing (must sum to 1).

    Returns:
        r_hat, c_hat: Smoothed marginals.
    """
    eps = eps.clamp(max=1.0)
    if eps.dim() == 0:
        eps = eps.unsqueeze(0)
    eps = eps.unsqueeze(-1)

    r_hat = (1 - w_r * eps) * r + w_r * eps / r.size(-1)
    c_hat = (1 - w_c * eps) * c + w_c * eps / c.size(-1)

    return r_hat, c_hat


def _batched_mdot(
    r: th.Tensor,
    c: th.Tensor,
    C: th.Tensor,
    gamma_f: float,
    gamma_i: float = 16,
    p: float = 1.5,
    q: float = 2.0,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, int, Dict[str, Any]]:
    """
    Batched MDOT (Mirror Descent Optimal Transport) solver.

    Solves multiple entropic-regularized OT problems simultaneously using
    temperature annealing with truncated Newton projections.

    Args:
        r: (batch, n) row marginals.
        c: (batch, m) column marginals.
        C: (n, m) or (batch, n, m) cost matrix.
        gamma_f: Final temperature (inverse regularization weight).
        gamma_i: Initial temperature.
        p: Exponent for the epsilon schedule.
        q: Temperature annealing factor.

    Returns:
        u: (batch, n) optimal row dual variables.
        v: (batch, m) optimal column dual variables.
        gamma_final: (batch,) final temperature achieved per problem.
        k_total: Total number of primitive operations.
        logs: Optimization logs.
    """
    batch_size = r.shape[0]
    device = r.device
    dtype = r.dtype

    projector = BatchedTruncatedNewtonProjector(device=device, dtype=dtype)

    # Compute entropy bounds for epsilon schedule
    H_r = -(r * (r + 1e-30).log()).sum(-1)
    H_c = -(c * (c + 1e-30).log()).sum(-1)
    H_min = th.min(H_r, H_c)
    eps_fn = lambda g_: H_min / (g_**p)

    logs = {"proj_logs": [], "gammas": []}

    gamma = min(gamma_i, gamma_f)
    gamma_per_problem = th.full((batch_size,), gamma, device=device, dtype=dtype)
    gamma_prev = th.zeros((batch_size,), device=device, dtype=dtype)
    active_mask = th.ones(batch_size, device=device, dtype=th.bool)

    # Initialize dual variables
    eps_d = eps_fn(gamma)
    r_hat, c_hat = _batched_smooth_marginals(r, c, eps_d / 2, w_r=0.9, w_c=0.1)
    u_init = r_hat.log()
    v_init = c_hat.log()
    u_cur = u_init.clone()
    v_cur = v_init.clone()
    u_prev = u_cur.clone()
    v_prev = v_cur.clone()

    t = 1
    max_outer_iter = 50
    done_all = False

    while active_mask.any() and t < max_outer_iter and not done_all:
        done = th.abs(gamma_per_problem - gamma_f) < 1e-5
        done_all = (done | ~active_mask).all()

        eps_d = eps_fn(gamma_per_problem)
        r_hat, c_hat = _batched_smooth_marginals(r, c, eps_d / 2, w_r=0.9, w_c=0.1)

        # Scale cost matrix by per-problem gamma
        if C.dim() == 2:
            gamma_C = gamma_per_problem.unsqueeze(-1).unsqueeze(-1) * C.unsqueeze(0)
        else:
            gamma_C = gamma_per_problem.unsqueeze(-1).unsqueeze(-1) * C

        # Save previous values for warm-starting
        u_prev = th.where(active_mask.unsqueeze(-1), u_cur.clone(), u_prev)
        v_prev = th.where(active_mask.unsqueeze(-1), v_cur.clone(), v_prev)

        # Project using warm-started initial values
        u_new, v_new, proj_log, success = projector.project(
            gamma_C, r_hat.log(), c_hat.log(), eps_d / 2, u_init, v_init, active_mask
        )

        u_cur = th.where(active_mask.unsqueeze(-1), u_new, u_cur)
        v_cur = th.where(active_mask.unsqueeze(-1), v_new, v_cur)

        logs["proj_logs"].append(proj_log)

        # Store previous gamma for warm-starting
        gamma_prev_old = gamma_prev.clone()
        gamma_prev = gamma_per_problem.clone()

        # Update gamma for non-converged problems
        gamma_per_problem = th.where(
            active_mask & ~done, th.clamp(gamma_per_problem * q, max=gamma_f), gamma_per_problem
        )

        # Warm-start initialization for next iteration (extrapolation)
        # Uses linear extrapolation from the previous two iterates, similar to the
        # unbatched solver in mdot.py. The extrapolation factor is clamped to [-2, 2]
        # to prevent instability when gamma changes rapidly between iterations.
        if not done_all:
            # Avoid division by zero for first iteration (gamma_prev_old starts at 0)
            denom = (gamma_prev - gamma_prev_old).clamp(min=1e-10)
            extrap_factor = ((gamma_per_problem - gamma_prev) / denom).unsqueeze(-1)
            extrap_factor = extrap_factor.clamp(-2.0, 2.0)
            u_init = th.where(
                active_mask.unsqueeze(-1) & (t > 1), u_cur + (u_cur - u_prev) * extrap_factor, u_cur
            )
            v_init = th.where(
                active_mask.unsqueeze(-1) & (t > 1), v_cur + (v_cur - v_prev) * extrap_factor, v_cur
            )

        logs["gammas"].append(gamma_per_problem.clone())
        t += 1

    k_total = sum([log["n_iter"] for log in logs["proj_logs"]])
    logs["success"] = active_mask
    logs["outer_iterations"] = t - 1

    return u_cur, v_cur, gamma_per_problem, k_total, logs


def _batched_round(P: th.Tensor, r: th.Tensor, c: th.Tensor) -> th.Tensor:
    """
    Batched Altschuler rounding to project onto feasible transport plans.

    Args:
        P: (batch, n, m) approximate transport plans.
        r: (batch, n) row marginals.
        c: (batch, m) column marginals.

    Returns:
        P_rounded: (batch, n, m) feasible transport plans in U(r, c).
    """
    # Scale rows
    row_sums = P.sum(-1)
    X = th.clamp(r / row_sums.clamp(min=1e-30), max=1.0)
    P = P * X.unsqueeze(-1)

    # Scale columns
    col_sums = P.sum(-2)
    Y = th.clamp(c / col_sums.clamp(min=1e-30), max=1.0)
    P = P * Y.unsqueeze(-2)

    # Fix remaining error with rank-1 correction
    err_r = (r - P.sum(-1)).clamp(min=0)
    err_c = (c - P.sum(-2)).clamp(min=0)
    err_r_norm = err_r.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1) + 1e-30
    P = P + err_r.unsqueeze(-1) * err_c.unsqueeze(-2) / err_r_norm

    return P


def _batched_rounded_cost(
    u: th.Tensor, v: th.Tensor, r: th.Tensor, c: th.Tensor, C: th.Tensor, gamma: th.Tensor
) -> th.Tensor:
    """
    Compute transport cost with rounding in log-domain (memory efficient).

    This avoids materializing the full n×m transport plan for each problem.

    Args:
        u: (batch, n) row dual variables.
        v: (batch, m) column dual variables.
        r: (batch, n) row marginals.
        c: (batch, m) column marginals.
        C: (n, m) or (batch, n, m) cost matrix.
        gamma: (batch,) temperature per problem.

    Returns:
        costs: (batch,) optimal transport costs.
    """
    batch_size = u.shape[0]

    if C.dim() == 2:
        C = C.unsqueeze(0).expand(batch_size, -1, -1)

    gamma = gamma.unsqueeze(-1).unsqueeze(-1)

    # Row rounding in log domain
    r_P_log = u + th.logsumexp(v.unsqueeze(-2) - gamma * C, dim=-1)
    delta_u = th.clamp(r.log() - r_P_log, max=0)
    u = u + delta_u

    # Column rounding in log domain
    c_P_log = v + th.logsumexp(u.unsqueeze(-1) - gamma * C, dim=-2)
    delta_v = th.clamp(c.log() - c_P_log, max=0)
    v = v + delta_v

    # Compute row error for rank-1 correction
    r_P_log = u + th.logsumexp(v.unsqueeze(-2) - gamma * C, dim=-1)
    r_P = r_P_log.exp()
    err_r = r - r_P
    err_r_normalized = err_r / (err_r.abs().sum(-1, keepdim=True) + 1e-30)

    # Column marginal after rounding
    c_P_log = v + th.logsumexp(u.unsqueeze(-1) - gamma * C, dim=-2)
    c_P = c_P_log.exp()
    err_c = c - c_P

    # Main cost term (in log domain for stability)
    log_P = u.unsqueeze(-1) + v.unsqueeze(-2) - gamma * C
    cost_main = th.logsumexp(log_P + C.log().clamp(min=-30), dim=(-1, -2)).exp()

    # Rank-1 correction term
    cost_correction = (
        (err_r_normalized.unsqueeze(-2) @ C @ err_c.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )

    return cost_main + cost_correction


def solve_OT_batched(
    r: th.Tensor,
    c: th.Tensor,
    C: th.Tensor,
    gamma_f: float = 1024.0,
    drop_tiny: bool = False,
    return_plan: bool = False,
    round: bool = True,
    log: bool = False,
) -> Union[th.Tensor, Tuple[th.Tensor, Dict[str, Any]]]:
    """
    Solve multiple entropic-regularized optimal transport problems in a single batched call.

    This function provides significant speedup (5-10x) over solving problems sequentially
    by amortizing GPU synchronization overhead across all problems in the batch.

    Args:
        r: (batch, n) row marginals. Each row must sum to 1.
        c: (batch, m) column marginals. Each row must sum to 1.
        C: Cost matrix. Either (n, m) for shared cost across all problems,
           or (batch, n, m) for per-problem costs. Recommended to scale to [0, 1].
        gamma_f: Temperature (inverse of regularization weight). Higher values give
                 more accurate solutions but take longer. Stable up to ~2^18 with float64.
        drop_tiny: Not supported in batched solver. Raises NotImplementedError if True.
        return_plan: If True, return transport plans instead of costs.
        round: If True, apply Altschuler rounding for feasible solutions.
        log: If True, also return optimization logs.

    Returns:
        If return_plan is False: (batch,) tensor of transport costs.
        If return_plan is True: (batch, n, m) tensor of transport plans.
        If log is True: tuple of (result, logs_dict).

    Example:
        >>> # Solve 32 OT problems of size 512×512
        >>> r = torch.rand(32, 512, device='cuda', dtype=torch.float64)
        >>> r = r / r.sum(-1, keepdim=True)
        >>> c = torch.rand(32, 512, device='cuda', dtype=torch.float64)
        >>> c = c / c.sum(-1, keepdim=True)
        >>> C = torch.rand(512, 512, device='cuda', dtype=torch.float64)
        >>> costs = solve_OT_batched(r, c, C, gamma_f=1024.)
    """
    # Input validation
    if r.dim() != 2:
        raise ValueError(f"r must be 2D (batch, n), got shape {r.shape}")
    if c.dim() != 2:
        raise ValueError(f"c must be 2D (batch, m), got shape {c.shape}")
    if C.dim() not in [2, 3]:
        raise ValueError(f"C must be 2D (n, m) or 3D (batch, n, m), got shape {C.shape}")
    if r.shape[0] != c.shape[0]:
        raise ValueError(f"Batch size mismatch: r has {r.shape[0]}, c has {c.shape[0]}")
    if C.dim() == 3 and C.shape[0] != r.shape[0]:
        raise ValueError(f"Batch size mismatch: C has {C.shape[0]}, r has {r.shape[0]}")

    if drop_tiny:
        raise NotImplementedError(
            "drop_tiny is not yet implemented for batched solver. "
            "Use solve_OT with drop_tiny=True for individual problems instead."
        )

    dtype = r.dtype

    # Use double precision for high gamma
    if gamma_f > 2**10 and dtype != th.float64:
        warnings.warn(
            f"Switching to float64 for gamma_f > 2^10. Output will be converted back to {dtype}."
        )
        r, c, C = r.double(), c.double(), C.double()

    # Solve
    u, v, gamma_final, k_total, opt_logs = _batched_mdot(r, c, C, gamma_f)

    # Convert back to original dtype
    u, v = u.to(dtype), v.to(dtype)
    gamma_final = gamma_final.to(dtype)
    if C.dtype != dtype:
        C = C.to(dtype)

    opt_logs["k_total"] = k_total

    if return_plan:
        # Expand C for broadcasting if shared
        C_expanded = C.unsqueeze(0) if C.dim() == 2 else C
        gamma_for_plan = gamma_final.unsqueeze(-1).unsqueeze(-1)

        P = (u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_for_plan * C_expanded).exp()
        if round:
            P = _batched_round(P, r, c)

        return (P, opt_logs) if log else P
    else:
        if round:
            costs = _batched_rounded_cost(u, v, r, c, C, gamma_final)
        else:
            C_expanded = C.unsqueeze(0) if C.dim() == 2 else C
            gamma_for_plan = gamma_final.unsqueeze(-1).unsqueeze(-1)
            P = (u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_for_plan * C_expanded).exp()
            costs = (P * C_expanded).sum(dim=(-2, -1))

        return (costs, opt_logs) if log else costs
