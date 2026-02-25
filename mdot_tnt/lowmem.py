"""
Low-memory MDOT-TNT solver using column-batched materialization.

Achieves O(nk + (n+m)d) working memory by:
  1. Never materializing the full n x m cost matrix C — cost blocks C[:, s:e]
     are computed on-the-fly from point clouds (X, Y) and a cost function.
  2. Never materializing the full n x m transport plan P — plan blocks are
     computed on-the-fly from dual variables and cost blocks.

The user controls the trade-off via block_size (k):
  - k = m: O(nm) memory, fastest (same as standard algorithm)
  - k = sqrt(m): O(n*sqrt(m)) memory (good trade-off)
  - k = 1: O(n) memory, slowest (like element-wise PyKeOps)

Supports two input modes:
  - Point cloud mode: X (n, d), Y (m, d), cost_fn — truly O(nk + (n+m)d)
  - Dense matrix mode: C (n, m) — O(nm) for C storage, O(nk) working memory

Based on:
    "A Truncated Newton Method for Optimal Transport"
    by Mete Kemertas, Amir-massoud Farahmand, Allan D. Jepson (ICLR, 2025).
"""

import math
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch as th

from mdot_tnt.mdot import adjust_schedule, smooth_marginals
from mdot_tnt.rounding import round_altschuler

# Type alias: (start, end) -> C[:, start:end] of shape (n, end - start)
CostBlockFn = Callable[[int, int], th.Tensor]


# ---------------------------------------------------------------------------
# Cost functions for point clouds
# ---------------------------------------------------------------------------


def squared_euclidean(X: th.Tensor, Y_block: th.Tensor) -> th.Tensor:
    """
    Squared Euclidean cost: C[i,j] = ||X[i] - Y[j]||^2.

    Args:
        X: Source points, shape (n, d).
        Y_block: Target point block, shape (k, d).

    Returns:
        Cost block of shape (n, k).
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    XX = (X * X).sum(-1, keepdim=True)  # (n, 1)
    YY = (Y_block * Y_block).sum(-1).unsqueeze(0)  # (1, k)
    return XX + YY - 2.0 * (X @ Y_block.T)


def euclidean(X: th.Tensor, Y_block: th.Tensor) -> th.Tensor:
    """
    Euclidean cost: C[i,j] = ||X[i] - Y[j]||.

    Args:
        X: Source points, shape (n, d).
        Y_block: Target point block, shape (k, d).

    Returns:
        Cost block of shape (n, k).
    """
    return squared_euclidean(X, Y_block).clamp(min=0.0).sqrt()


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class LowMemoryTruncatedNewtonProjector:
    """
    Low-memory variant of the Truncated Newton projector.

    Instead of storing the full cost matrix or transport plan, cost blocks
    are computed on-the-fly via a user-supplied ``cost_block_fn``.  All LSE
    reductions, Hessian-vector products, and diagonal preconditioner
    computations are performed block-by-block with O(nk) working memory.

    Args:
        device: PyTorch device for computations.
        dtype: Data type for tensors.
        block_size: Number of columns to process at once.
        cost_block_fn: ``(start, end) -> C[:, start:end]`` of shape ``(n, end-start)``.
        **kwargs: Additional options (debug: bool for verbose output).
    """

    def __init__(
        self,
        device: th.device,
        dtype: th.dtype,
        block_size: int,
        cost_block_fn: CostBlockFn,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.cost_block_fn = cost_block_fn
        self.gamma: Union[float, th.Tensor] = 1.0
        self.rho = th.zeros(1, device=device, dtype=dtype)
        self.debug = kwargs.get("debug", False)

    # -- helpers -------------------------------------------------------------

    def _col_blocks(self, m: int):
        """Yield (start, end) index pairs for column blocks."""
        for start in range(0, m, self.block_size):
            yield start, min(start + self.block_size, m)

    def _gamma_C_block(self, s: int, e: int) -> th.Tensor:
        """Return gamma * C[:, s:e], shape (n, e-s)."""
        return self.gamma * self.cost_block_fn(s, e)

    def _P_block(self, u: th.Tensor, v: th.Tensor, s: int, e: int) -> th.Tensor:
        """Compute P[:, s:e] on-the-fly from dual variables and cost block."""
        return th.exp(u.unsqueeze(-1) + v[s:e].unsqueeze(-2) - self._gamma_C_block(s, e))

    # -- blocked primitives --------------------------------------------------

    def _blocked_lse_r(self, v: th.Tensor) -> th.Tensor:
        """
        logsumexp(v[None, :] - gamma*C, dim=-1) via streaming logaddexp.

        Returns shape (n,).
        """
        m = v.shape[-1]
        result: Optional[th.Tensor] = None
        for s, e in self._col_blocks(m):
            partial = th.logsumexp(v[s:e].unsqueeze(-2) - self._gamma_C_block(s, e), dim=-1)
            if result is None:
                result = partial
            else:
                result = th.logaddexp(result, partial)
        assert result is not None
        return result

    def _blocked_lse_c(self, u: th.Tensor, m: int) -> th.Tensor:
        """
        logsumexp(u[:, None] - gamma*C, dim=-2) via column blocks.

        Each block of k columns produces k output elements (reduction is over
        rows), so results are concatenated directly.

        Returns shape (m,).
        """
        results = []
        for s, e in self._col_blocks(m):
            results.append(th.logsumexp(u.unsqueeze(-1) - self._gamma_C_block(s, e), dim=-2))
        return th.cat(results, dim=-1)

    def _blocked_diag_PPc(
        self,
        u: th.Tensor,
        v: th.Tensor,
        c: th.Tensor,
    ) -> th.Tensor:
        """
        Compute diag(P diag(1/c) P^T) = ((P**2) / c[None, :]).sum(-1)
        in column blocks.

        Returns shape (n,).
        """
        m = v.shape[-1]
        result = th.zeros_like(u)
        for s, e in self._col_blocks(m):
            Pb = self._P_block(u, v, s, e)
            result += ((Pb**2) / c[s:e].unsqueeze(-2)).sum(-1)
            del Pb
        return result

    def _blocked_PPc_matmul(
        self,
        u: th.Tensor,
        v: th.Tensor,
        c: th.Tensor,
        x: th.Tensor,
    ) -> th.Tensor:
        """
        P diag(1/c) P^T x in a single pass over column blocks.

        Uses the decomposition:
            P diag(1/c) P^T x = sum_b P_b diag(1/c_b) P_b^T x

        Returns shape (n,).
        """
        m = v.shape[-1]
        result = th.zeros_like(x)
        for s, e in self._col_blocks(m):
            Pb = self._P_block(u, v, s, e)
            z = (x @ Pb) / c[s:e]  # P_b^T x / c_b, shape (k,)
            result += Pb @ z  # P_b z, shape (n,)
            del Pb, z
        return result

    def _blocked_Pc_x(
        self,
        u: th.Tensor,
        v: th.Tensor,
        c: th.Tensor,
        x: th.Tensor,
    ) -> th.Tensor:
        """
        diag(1/c) P^T x in column blocks.

        Returns shape (m,).
        """
        m = v.shape[-1]
        result = th.empty(m, device=self.device, dtype=self.dtype)
        for s, e in self._col_blocks(m):
            Pb = self._P_block(u, v, s, e)
            result[s:e] = (x @ Pb) / c[s:e]
            del Pb
        return result

    # -- main projection -----------------------------------------------------

    def project(
        self,
        gamma: Union[float, th.Tensor],
        log_r: th.Tensor,
        log_c: th.Tensor,
        eps_d: Union[float, th.Tensor],
        u: th.Tensor,
        v: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Any], bool]:
        """
        Project onto the set of couplings satisfying marginal constraints.

        Args:
            gamma: Temperature (inverse regularization weight) for this
                projection step.
            log_r: Log of row marginals, shape (n,).
            log_c: Log of column marginals, shape (m,).
            eps_d: Convergence tolerance for the dual gradient norm.
            u: Initial row dual variables, shape (n,).
            v: Initial column dual variables, shape (m,).

        Returns:
            u: Updated row dual variables.
            v: Updated column dual variables.
            logs: Dictionary with optimization statistics.
            success: Whether projection converged successfully.
        """
        self.gamma = gamma
        m = v.shape[-1]

        logs: Dict[str, Any] = {
            "errs": [],
            "ls_func_cnt": 0,
            "chisinkhorn_steps": 0,
            "newtonsolve_steps": 0,
            "deltas": [],
            "all_newtonsolve_steps": [],
        }
        success_fn = lambda err_: err_ < 10 * eps_d

        r = log_r.exp()
        c = log_c.exp()

        # Blocked LSE operations — O(nk) working memory per call.
        self.LSE_r = lambda v_: self._blocked_lse_r(v_)
        self.LSE_c = lambda u_: self._blocked_lse_c(u_, m)

        log_c_P = v + self.LSE_c(u)
        v += log_c - log_c_P
        log_r_P = u + self.LSE_r(v)
        k = 8

        u, v, log_r_P, err, k_ = self.chi_sinkhorn(u, v, log_r, log_c, log_r_P, eps_d ** (2 / 5))
        r_P = log_r_P.exp()
        logs["errs"].append(err)
        logs["chisinkhorn_steps"] = k_
        k += k_

        num_iter = 0

        while err > eps_d:
            num_iter += 1

            beta = 0.5
            eta_k = th.max(err, 0.9 * (eps_d / err))

            grad_k = r_P - r
            self.rho = th.max(th.zeros_like(self.rho), self.rho)

            diag_PPc = self._blocked_diag_PPc(u, v, c)
            k += 8

            delta_u, delta_v, matmul_cnt, rho, pcg_success = self.newton_solve(
                u,
                v,
                c,
                diag_PPc,
                grad_k,
                r_P,
                err,
                beta,
                eta_k,
                maxIter=5000,
            )
            if not pcg_success:
                k += matmul_cnt
                logs["n_iter"] = k
                msg = (
                    "PCG did not converge. TruncatedNewton returning with "
                    f"success={success_fn(err)}"
                )
                warnings.warn(msg)
                return u, v, logs, success_fn(err)

            self.rho = th.max(th.zeros_like(self.rho), 1.0 - (1.0 - rho) * 4.0)
            k += matmul_cnt
            logs["newtonsolve_steps"] += matmul_cnt

            alpha = th.ones_like(self.rho)
            log_c_P = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
            k += 4
            linear_decr = -(grad_k * delta_u).sum(-1, keepdim=True)
            if not linear_decr > 0:
                logs["n_iter"] = k
                msg = (
                    "Linear decrease condition not satisfied. TruncatedNewton "
                    f"returning with success={success_fn(err)}"
                )
                warnings.warn(msg)
                return u, v, logs, success_fn(err)

            armijo = log_c_P.exp().sum(-1, keepdim=True) - 1 <= 0.99 * alpha * linear_decr
            while not armijo:
                alpha *= 0.5
                if alpha < 1e-9:
                    logs["n_iter"] = k
                    msg = (
                        "Line search did not converge. TruncatedNewton "
                        f"returning with success={success_fn(err)}"
                    )
                    warnings.warn(msg)
                    return u, v, logs, success_fn(err)

                log_c_P = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
                k += 4
                logs["ls_func_cnt"] += 4
                armijo = log_c_P.exp().sum(-1, keepdim=True) - 1 <= 0.99 * alpha * linear_decr

            u += alpha * delta_u
            v += alpha * delta_v

            err_before_sk = (c - log_c_P.exp()).abs().sum(-1)
            err_before_sk += (r - (u + self.LSE_r(v)).exp()).abs().sum(-1)

            v += log_c - log_c_P

            log_r_P = u + self.LSE_r(v)
            k += 4

            u, v, log_r_P, err, k_ = self.chi_sinkhorn(
                u, v, log_r, log_c, log_r_P, eps_d ** (2 / 5)
            )
            r_P = log_r_P.exp()
            logs["chisinkhorn_steps"] += k_
            k += k_

            logs["errs"].append(err)
            logs["deltas"].append(
                th.min((logs["errs"][-2] - err_before_sk) / ((1 - eta_k) * logs["errs"][-2])).item()
            )

        if u.isnan().any() or v.isnan().any():
            raise ValueError("NaNs encountered in u or v")

        logs["n_iter"] = k

        delta_u = log_r - log_r_P
        u += delta_u

        return u, v, logs, True

    # -- chi-squared Sinkhorn ------------------------------------------------

    def chi_sinkhorn(
        self,
        u: th.Tensor,
        v: th.Tensor,
        log_r: th.Tensor,
        log_c: th.Tensor,
        log_r_P: th.Tensor,
        eps_chi: Union[float, th.Tensor],
        maxOps: float = float("inf"),
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, int]:
        """Chi-squared Sinkhorn iterations using blocked LSE."""
        k = 0
        r = log_r.exp()
        err = (r - log_r_P.exp()).norm(p=1, dim=-1)
        r_P = log_r_P.exp()
        chi_squared = ((r - r_P) ** 2 / r_P).sum(-1)

        while chi_squared > eps_chi and k < maxOps:
            u += log_r - log_r_P

            log_c_P = v + self.LSE_c(u)
            v += log_c - log_c_P

            log_r_P = u + self.LSE_r(v)
            r_P = log_r_P.exp()

            err = (r - r_P).norm(p=1, dim=-1)
            chi_squared = ((r - r_P) ** 2 / r_P).sum(-1)
            k += 8

        if k >= maxOps:
            raise ValueError(f"Chi-Sinkhorn did not converge in maxIter={maxOps} steps")

        return u, v, log_r_P, err, k

    # -- Newton solve --------------------------------------------------------

    def newton_solve(
        self,
        u: th.Tensor,
        v: th.Tensor,
        c: th.Tensor,
        diag_PPc: th.Tensor,
        grad_k: th.Tensor,
        r_P: th.Tensor,
        err: th.Tensor,
        beta: float = 0.5,
        eta_k: Union[float, th.Tensor] = 0.5,
        maxIter: int = 500,
    ) -> Tuple[th.Tensor, th.Tensor, int, th.Tensor, bool]:
        """
        Newton solve using blocked Hessian-vector products.

        Transport plan blocks are recomputed on-the-fly for each mat-vec
        product from the (unchanged) dual variables u, v and the cost function.
        """
        rho = self.rho
        tol = err * eta_k

        def mml(x_):
            return self._blocked_PPc_matmul(u, v, c, x_)

        M = lambda rho_: r_P - rho_ * diag_PPc
        M_rho = M(th.ones_like(self.rho))
        M_rho[M_rho <= 0] = M_rho[M_rho > 0].min()

        x0 = -grad_k / M_rho
        PPc_x0 = mml(x0)
        matmul_cnt = 2
        r_P_x0 = r_P * x0

        x = x0.clone()
        PPc_x = PPc_x0.clone()
        r_P_x = r_P_x0.clone()

        res_true = r_P_x0 - PPc_x + grad_k

        linear_decr = (x * -grad_k).sum(-1)
        if linear_decr <= 0:
            raise ValueError("Linear decrease condition not satisfied")

        r_true_norm = res_true.norm(p=1, dim=-1)
        best_sol = x.clone()
        best_r_true_norm = r_true_norm.clone()

        done = False
        success = True

        while best_r_true_norm > tol:
            best_sol[r_true_norm < best_r_true_norm] = x[r_true_norm < best_r_true_norm]
            best_r_true_norm = th.min(r_true_norm, best_r_true_norm)

            rho[r_true_norm > tol] = 1.0 - (1.0 - rho[r_true_norm > tol]) * 0.25
            M_rho = M(rho)

            if matmul_cnt > 0:
                x = x0.clone()
                PPc_x = PPc_x0.clone()
                r_P_x = r_P_x0.clone()

            Fr_x = r_P_x - rho * PPc_x
            res = Fr_x + grad_k

            res_true = r_P_x - PPc_x + grad_k
            r_true_norm = res_true.norm(p=1, dim=-1)
            best_r_true_norm = th.min(r_true_norm, best_r_true_norm)
            linear_decr = (x * -grad_k).sum(-1)
            if (best_r_true_norm < tol).all() and (linear_decr > 0).all():
                break

            y = res / M_rho
            p = -y.clone()
            ry_old = (res * y).sum(-1, keepdim=True)

            r_norm = res.norm(p=1, dim=-1)
            while (r_norm > 0.5 * (1 - beta) * tol)[best_r_true_norm > tol].any():
                PPc_p = mml(p)
                matmul_cnt += 2
                Fr_p = (r_P * p) - rho * PPc_p

                quad = (Fr_p * p).sum(-1, keepdim=True)
                if (quad <= 0)[best_r_true_norm > tol].any():
                    warnings.warn(
                        "Warning: negative curvature encountered in CG. "
                        "Returning best solution. Residual norm less than "
                        f"error: {(best_r_true_norm < err).item()}"
                    )
                    x = best_sol.clone()
                    done = True
                    success = best_r_true_norm < err
                    warnings.warn("Resetting discount factor rho = 0")
                    rho = th.zeros_like(self.rho)
                    break

                alpha = ry_old / quad
                x += alpha * p
                res += alpha * Fr_p
                r_norm = res.norm(p=1, dim=-1)

                if (
                    th.isnan(r_norm)[best_r_true_norm > tol].any()
                    or th.isinf(r_norm)[best_r_true_norm > tol].any()
                ):
                    raise ValueError("NaNs or infs encountered in r_norm")

                PPc_x += alpha * PPc_p

                r_P_x = r_P * x
                res_true = r_P_x - PPc_x + grad_k
                r_true_norm = res_true.norm(p=1, dim=-1)
                best_sol[r_true_norm < best_r_true_norm] = x[r_true_norm < best_r_true_norm]
                best_r_true_norm = th.min(r_true_norm, best_r_true_norm)

                linear_decr = (x * -grad_k).sum(-1)
                if (best_r_true_norm <= tol).all() and (linear_decr > 0).all():
                    done = True
                    success = True
                    break

                if matmul_cnt > 2 * maxIter:
                    warnings.warn("PCG did not converge.")
                    done = True
                    success = False
                    break

                y = res / M_rho
                ry_new = (res * y).sum(-1, keepdim=True)
                p = -y + (ry_new / ry_old) * p
                ry_old = ry_new.clone()

            if done:
                break

        if r_true_norm <= tol:
            success = True

        x = best_sol
        Pc_x = self._blocked_Pc_x(u, v, c, x)
        matmul_cnt += 1

        return x, -Pc_x, matmul_cnt, rho, success


# ---------------------------------------------------------------------------
# Blocked cost computation
# ---------------------------------------------------------------------------


def blocked_rounded_cost(
    u: th.Tensor,
    v: th.Tensor,
    r: th.Tensor,
    c: th.Tensor,
    cost_block_fn: CostBlockFn,
    m: int,
    gamma: Union[float, th.Tensor],
    block_size: int,
) -> th.Tensor:
    """
    Compute the rounded transport cost without materializing C or P.

    Mirrors ``rounded_cost_altschuler`` but accesses the cost matrix only
    through ``cost_block_fn``.  Working memory is O(n * block_size).

    Args:
        u: Dual variable for rows, shape (n,).
        v: Dual variable for columns, shape (m,).
        r: Row marginal, shape (n,).
        c: Column marginal, shape (m,).
        cost_block_fn: ``(s, e) -> C[:, s:e]``.
        m: Number of columns.
        gamma: Temperature.
        block_size: Number of columns per block.

    Returns:
        Transport cost (scalar tensor).
    """

    def col_blocks():
        for s in range(0, m, block_size):
            yield s, min(s + block_size, m)

    def blocked_lse_r():
        result = None
        for s, e in col_blocks():
            Cb = cost_block_fn(s, e)
            partial = th.logsumexp(v[s:e].unsqueeze(-2) - gamma * Cb, dim=-1)
            if result is None:
                result = partial
            else:
                result = th.logaddexp(result, partial)
        return result

    def blocked_lse_c():
        parts = []
        for s, e in col_blocks():
            Cb = cost_block_fn(s, e)
            parts.append(th.logsumexp(u.unsqueeze(-1) - gamma * Cb, dim=-2))
        return th.cat(parts, dim=-1)

    # Step 1: Row rounding
    r_P_log = u + blocked_lse_r()
    delta_u = th.min(r.log() - r_P_log, th.zeros_like(r))
    u = u + delta_u

    # Step 2: Column rounding
    c_P_log = v + blocked_lse_c()
    delta_v = th.min(c.log() - c_P_log, th.zeros_like(c))
    v = v + delta_v

    # Step 3: Residual errors for rank-1 correction
    r_P_log = u + blocked_lse_r()
    r_P = r_P_log.exp()
    err_r = r - r_P
    err_r = err_r / (err_r.norm(p=1, dim=-1, keepdim=True) + 1e-30)

    c_P_log = v + blocked_lse_c()
    c_P = c_P_log.exp()
    err_c = c - c_P

    # Step 4: cost = sum_{i,j} P_{ij} C_{ij}
    #       = sum_{i,j} exp(u_i + v_j - gamma*C_{ij}) * C_{ij}
    log_cost: Optional[th.Tensor] = None
    for s, e in col_blocks():
        Cb = cost_block_fn(s, e)
        log_block = u.unsqueeze(-1) + v[s:e].unsqueeze(-2) - gamma * Cb + Cb.log()
        del Cb
        partial = th.logsumexp(log_block.reshape(-1), dim=0)
        del log_block
        if log_cost is None:
            log_cost = partial
        else:
            log_cost = th.logaddexp(log_cost, partial)
    assert log_cost is not None
    cost = log_cost.exp()

    # Step 5: Rank-1 correction: err_r^T C err_c
    correction = th.zeros(1, device=u.device, dtype=u.dtype)
    for s, e in col_blocks():
        Cb = cost_block_fn(s, e)
        correction += (err_r @ Cb) @ err_c[s:e]
        del Cb
    cost += correction.squeeze()

    return cost


def blocked_transport_cost(
    u: th.Tensor,
    v: th.Tensor,
    cost_block_fn: CostBlockFn,
    m: int,
    gamma: Union[float, th.Tensor],
    block_size: int,
) -> th.Tensor:
    """
    Compute unrounded transport cost without materializing C or P.

    cost = sum_{i,j} exp(u_i + v_j - gamma * C_{ij}) * C_{ij}

    Args:
        u: Dual variable for rows, shape (n,).
        v: Dual variable for columns, shape (m,).
        cost_block_fn: ``(s, e) -> C[:, s:e]``.
        m: Number of columns.
        gamma: Temperature.
        block_size: Number of columns per block.

    Returns:
        Transport cost (scalar tensor).
    """
    log_cost: Optional[th.Tensor] = None
    for s in range(0, m, block_size):
        e = min(s + block_size, m)
        Cb = cost_block_fn(s, e)
        log_block = u.unsqueeze(-1) + v[s:e].unsqueeze(-2) - gamma * Cb + Cb.log()
        del Cb
        partial = th.logsumexp(log_block.reshape(-1), dim=0)
        del log_block
        if log_cost is None:
            log_cost = partial
        else:
            log_cost = th.logaddexp(log_cost, partial)
    assert log_cost is not None
    return log_cost.exp()


# ---------------------------------------------------------------------------
# MDOT solver (low-memory variant)
# ---------------------------------------------------------------------------


def mdot_lowmem(
    r: th.Tensor,
    c: th.Tensor,
    cost_block_fn: CostBlockFn,
    gamma_f: float,
    block_size: int,
    gamma_i: float = 16,
    p: float = 1.5,
    q: float = 2.0,
) -> Tuple[th.Tensor, th.Tensor, float, int, Dict[str, Any]]:
    """
    Solve entropic-regularized OT using MDOT with low-memory projection.

    Args:
        r: Row marginal, shape (n,).
        c: Column marginal, shape (m,).
        cost_block_fn: ``(s, e) -> C[:, s:e]`` — computes cost blocks lazily.
        gamma_f: Final temperature (inverse regularization weight).
        block_size: Number of columns per block for the projector.
        gamma_i: Initial temperature.
        p: Exponent for the epsilon schedule.
        q: Temperature annealing factor.

    Returns:
        u: Row dual variables, shape (n,).
        v: Column dual variables, shape (m,).
        gamma: Final temperature achieved.
        k_total: Total number of O(n^2) primitive operations.
        logs: Optimization statistics.
    """
    projector = LowMemoryTruncatedNewtonProjector(
        device=r.device,
        dtype=r.dtype,
        block_size=block_size,
        cost_block_fn=cost_block_fn,
    )

    H_r = -(r * (r + 1e-30).log()).sum(-1)
    H_c = -(c * (c + 1e-30).log()).sum(-1)
    H_min = th.min(H_r, H_c)
    eps_fn = lambda g_: H_min / (g_**p)

    logs: Dict[str, Any] = {
        "proj_logs": [],
        "eps": [],
    }

    t = 1
    done = False
    gamma = min(gamma_i, gamma_f)
    gammas = [0.0, gamma]

    while not done:
        done = abs(gamma - gamma_f) < 1e-5

        eps_d = eps_fn(gamma)

        r_hat, c_hat = smooth_marginals(r, c, eps_d / 2, w_r=0.9, w_c=0.1)

        if t == 1:
            u_init, v_init = r_hat.log(), c_hat.log()
            u_cur, v_cur = u_init.clone(), v_init.clone()

        u_prev, v_prev = u_cur.clone(), v_cur.clone()

        u_cur, v_cur, proj_log, success = projector.project(
            gamma, r_hat.log(), c_hat.log(), eps_d / 2, u_init, v_init
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
            u_init = u_cur + (u_cur - u_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])
            v_init = v_cur + (v_cur - v_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])

        gammas.append(gamma)
        t += 1

    k_total = sum([log["n_iter"] for log in logs["proj_logs"]])
    k_total += t - 1
    logs["success"] = success
    logs["gammas"] = gammas

    return u_cur, v_cur, gammas[-1], k_total, logs


# ---------------------------------------------------------------------------
# Marginal preprocessing (inlined to avoid depending on a dense C)
# ---------------------------------------------------------------------------


def _preprocess_marginal(m: th.Tensor, eps: float) -> Tuple[th.Tensor, th.Tensor]:
    """Drop smallest entries whose cumulative mass is below ``eps``."""
    m_sorted, m_idx = th.sort(m, dim=-1, descending=False)
    m_cumsum = th.cumsum(m_sorted, dim=-1)
    m_keep = m_idx[m_cumsum > eps]
    m_new = m[m_keep]
    mass_removed = 1 - m_new.sum(-1)
    m_new = m_new + mass_removed / m_new.size(-1)
    return m_new, m_keep


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def solve_OT_lowmem(
    r: th.Tensor,
    c: th.Tensor,
    C: Optional[th.Tensor] = None,
    X: Optional[th.Tensor] = None,
    Y: Optional[th.Tensor] = None,
    cost_fn: Optional[Callable[[th.Tensor, th.Tensor], th.Tensor]] = None,
    gamma_f: float = 1024.0,
    block_size: Optional[int] = None,
    drop_tiny: bool = False,
    return_plan: bool = False,
    round: bool = True,
    log: bool = False,
):
    """
    Solve entropic-regularized OT with O(nk) working memory.

    Accepts either a dense cost matrix **or** point clouds with a cost
    function.  In point-cloud mode the cost matrix is never materialised,
    giving true O(nk + (n+m)d) memory.

    Args:
        r: Row marginal, shape (n,). Must sum to 1.
        c: Column marginal, shape (m,). Must sum to 1.
        C: Dense cost matrix, shape (n, m). Mutually exclusive with X/Y.
            Recommended to scale entries to [0, 1].
        X: Source points, shape (n, d). Use with Y and optionally cost_fn.
        Y: Target points, shape (m, d). Use with X.
        cost_fn: ``(X, Y_block) -> C_block`` of shape (n, k).  Defaults to
            ``squared_euclidean``.  Only used when X/Y are provided.
        gamma_f: Temperature (inverse regularization weight).
        block_size: Columns per block.  Controls memory / compute trade-off:
                    - None or m: fastest, O(nm) or O(nk) memory
                    - sqrt(m): good trade-off
                    - 1: minimum memory (slowest)
        drop_tiny: Drop tiny marginal entries for speedup with sparse
            marginals.
        return_plan: If True, return the full transport plan.  Note: the plan
            is inherently O(nm); in point-cloud mode it is built block-by-block
            so that C and P are never both in memory simultaneously.
        round: Use Altschuler rounding for feasible plans / costs.
        log: Additionally return optimization logs.

    Returns:
        Transport cost (scalar) if return_plan is False, or the transport plan
        (n, m) if return_plan is True.  If log is True, returns (result, logs).
    """
    # -- input validation ----------------------------------------------------
    have_C = C is not None
    have_XY = X is not None and Y is not None
    if not (have_C ^ have_XY):
        raise ValueError(
            "Provide exactly one of: C (dense cost matrix) or X and Y "
            f"(point clouds). Got C={C is not None}, X={X is not None}, Y={Y is not None}."
        )

    # -- dtype promotion -----------------------------------------------------
    dtype = r.dtype
    if gamma_f > 2**10 and dtype != th.float64:
        warnings.warn(
            "Switching to double precision for gamma_f > 2^10 during "
            f"execution. Output will be input dtype: {dtype}."
        )
        r, c = r.double(), c.double()
        if have_C:
            assert C is not None
            C = C.double()
        else:
            assert X is not None and Y is not None
            X, Y = X.double(), Y.double()

    # -- build cost_block_fn -------------------------------------------------
    cost_block_fn: CostBlockFn
    if have_C:
        assert C is not None
        m = C.shape[-1]
        cost_block_fn = lambda s, e: C[:, s:e]
    else:
        assert X is not None and Y is not None
        m = Y.shape[0]
        _cf = cost_fn if cost_fn is not None else squared_euclidean
        cost_block_fn = lambda s, e: _cf(X, Y[s:e])

    if block_size is None:
        block_size = m

    # -- optional marginal sparsification ------------------------------------
    if drop_tiny:
        drop_lessthan = math.log(min(r.size(-1), c.size(-1))) / (gamma_f**2)
        r_, r_keep = _preprocess_marginal(r, drop_lessthan)
        c_, c_keep = _preprocess_marginal(c, drop_lessthan)

        cbf_: CostBlockFn
        if have_C:
            assert C is not None
            C_ = C[r_keep][:, c_keep]
            cbf_ = lambda s, e: C_[:, s:e]
        else:
            assert X is not None and Y is not None
            X_ = X[r_keep]
            Y_ = Y[c_keep]
            cbf_ = lambda s, e: _cf(X_, Y_[s:e])

        u_, v_, gamma_f_, k_total, opt_logs = mdot_lowmem(r_, c_, cbf_, gamma_f, block_size)

        u = -th.ones_like(r) * float("inf")
        u[r_keep] = u_
        v = -th.ones_like(c) * float("inf")
        v[c_keep] = v_
    else:
        u, v, gamma_f_, k_total, opt_logs = mdot_lowmem(r, c, cost_block_fn, gamma_f, block_size)

    u, v = u.to(dtype), v.to(dtype)

    # -- build cost_block_fn at final gamma (for cost / plan) ----------------
    # If drop_tiny was used, cost_block_fn should operate on the full
    # (un-subsetted) dimensions — it was defined above before drop_tiny.

    if return_plan:
        # Build the plan block-by-block so C and P are never both in memory.
        blocks = []
        for s in range(0, m, block_size):
            e = min(s + block_size, m)
            Cb = cost_block_fn(s, e)
            Pb = th.exp(u.unsqueeze(-1) + v[s:e].unsqueeze(-2) - gamma_f_ * Cb)
            del Cb
            blocks.append(Pb)
        P = th.cat(blocks, dim=-1)
        del blocks
        if round:
            P = round_altschuler(P, r, c)
        return (P, opt_logs) if log else P
    else:
        if round:
            cost = blocked_rounded_cost(u, v, r, c, cost_block_fn, m, gamma_f_, block_size)
        else:
            cost = blocked_transport_cost(u, v, cost_block_fn, m, gamma_f_, block_size)
        return (cost, opt_logs) if log else cost
