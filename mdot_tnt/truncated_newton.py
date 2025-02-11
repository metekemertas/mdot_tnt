
import torch as th
import warnings


class TruncatedNewtonProjector:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.rho = th.zeros(1, device=device, dtype=dtype)
        self.debug = kwargs.get('debug', False)

    def project(self, gamma_C, log_r, log_c, eps_d, u, v):
        """
        Project onto the set of couplings that satisfy the marginal constraints.
        :param gamma_C: The cost matrix scaled by gamma.
        :param log_r:
        """
        logs = {
            "errs": [],
            'ls_func_cnt': 0,
            'chisinkhorn_steps': 0,
            'newtonsolve_steps': 0,
            "deltas": [],  # Ratios of actual to theoretically predicted (ideal) reduction in gradient norm.
            "all_newtonsolve_steps": []
        }
        # In case of errors or issues, 10 times the tolerance level is considered
        # a good enough solution to keep MDOT going.
        success_fn = lambda err_: err_ < 10 * eps_d

        r = log_r.exp()
        c = log_c.exp()

        # Each LSE operation costs 4 * n^2 operations.
        self.LSE_r = lambda v_: th.logsumexp(v_.unsqueeze(-2) - gamma_C, dim=-1)
        self.LSE_c = lambda u_: th.logsumexp(u_.unsqueeze(-1) - gamma_C, dim=-2)

        log_c_P = v + self.LSE_c(u)
        v += log_c - log_c_P  # Ensure c=c(P)
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

            P = th.exp(u.unsqueeze(-1) + v.unsqueeze(-2) - gamma_C)
            diag_PPc = ((P ** 2) / c.unsqueeze(-2)).sum(-1)
            k += 8
            delta_u, delta_v, matmul_cnt, rho, pcg_success = self.newton_solve(
                P, c, diag_PPc, grad_k, r_P, err, beta, eta_k, maxIter=5000)
            del P  # Free up memory
            if not pcg_success:
                k += matmul_cnt
                logs["n_iter"] = k
                msg = "PCG did not converge. TruncatedNewton returning with success={}".format(success_fn(err))
                warnings.warn(msg)
                return u, v, logs, success_fn(err)

            self.rho = th.max(th.zeros_like(self.rho), 1. - (1. - rho) * 4.)
            k += matmul_cnt
            logs["newtonsolve_steps"] += matmul_cnt

            alpha = th.ones_like(self.rho)
            log_c_P = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
            k += 4
            linear_decr = -(grad_k * delta_u).sum(-1, keepdim=True)
            if not linear_decr > 0:
                logs["n_iter"] = k
                msg = "Linear decrease condition not satisfied. TruncatedNewton returning with success={}".format(
                    success_fn(err))
                warnings.warn(msg)
                return u, v, logs, success_fn(err)

            armijo = log_c_P.exp().sum(-1, keepdim=True) - 1 <= 0.99 * alpha * linear_decr
            while not armijo:  # Check armijo condition for batch elements where err > eps_d
                alpha *= 0.5
                if alpha < 1e-9:
                    logs["n_iter"] = k
                    msg = "Line search did not converge. TruncatedNewton returning with success={}".format(
                        success_fn(err))
                    warnings.warn(msg)
                    return u, v, logs, success_fn(err)

                log_c_P = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
                k += 4
                logs["ls_func_cnt"] += 4
                armijo = log_c_P.exp().sum(-1, keepdim=True) - 1 <= 0.99 * alpha * linear_decr

            u += alpha * delta_u
            v += alpha * delta_v

            # The below error (before the Sinkhorn update) is used
            # to measure the progress of the algorithm with TruncatedNewton steps.
            err_before_sk = (c - log_c_P.exp()).abs().sum(-1)
            err_before_sk += (r - (u + self.LSE_r(v)).exp()).abs().sum(-1)

            # Sinkhorn update to ensure c=c(P).
            v += log_c - log_c_P

            log_r_P = u + self.LSE_r(v)
            k += 4

            u, v, log_r_P, err, k_ = self.chi_sinkhorn(u, v, log_r, log_c, log_r_P, eps_d ** (2 / 5))
            r_P = log_r_P.exp()
            logs["chisinkhorn_steps"] += k_
            k += k_

            logs["errs"].append(err)
            logs["deltas"].append(th.min((logs["errs"][-2] - err_before_sk) / ((1 - eta_k) * logs["errs"][-2])).item())

        if u.isnan().any() or v.isnan().any():
            raise ValueError("NaNs encountered in u or v")

        logs["n_iter"] = k

        # Since we already computed log_r_P, we can use it to perform one last Sinkhorn update on rows.
        delta_u = log_r - log_r_P
        u += delta_u

        return u, v, logs, True

    def chi_sinkhorn(self, u, v, log_r, log_c, log_r_P, eps_chi, maxOps=float('inf')):
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
            raise ValueError("Chi-Sinkhorn did not converge in maxIter={} steps".format(maxOps))

        return u, v, log_r_P, err, k

    def newton_solve(self, P, c, diag_PPc, grad_k, r_P, err, beta=0.5, eta_k=0.5, maxIter=500):
        rho = self.rho
        tol = err * eta_k

        matmul_PPc = lambda x_: (P @ ((x_.unsqueeze(-2) @ P).transpose(-2, -1) / c.unsqueeze(-1))).squeeze(-1)

        # mml = th.compile(matmul_PPc)
        mml = matmul_PPc

        M = lambda rho_: (r_P - rho_ * diag_PPc)  # Diagonal preconditioner
        M_rho = M(th.ones_like(self.rho))
        M_rho[M_rho <= 0] = M_rho[M_rho > 0].min()

        x0 = -grad_k / M_rho
        PPc_x0 = mml(x0)
        matmul_cnt = 2
        r_P_x0 = (r_P * x0)

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

            rho[r_true_norm > tol] = 1. - (1. - rho[r_true_norm > tol]) * 0.25
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
                    warnings.warn("Warning: negative curvature encountered in CG. Returning best solution. "
                          "Residual norm less than error: {}".format((best_r_true_norm < err).item()))
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

                if th.isnan(r_norm)[best_r_true_norm > tol].any() or th.isinf(r_norm)[best_r_true_norm > tol].any():
                    raise ValueError("NaNs or infs encountered in r_norm")

                PPc_x += alpha * PPc_p

                r_P_x = (r_P * x)
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
        Pc_x = ((x.unsqueeze(-2) @ P).transpose(-2, -1) / c.unsqueeze(-1)).squeeze(-1)
        matmul_cnt += 1

        return x, -Pc_x, matmul_cnt, rho, success