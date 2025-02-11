This is the official repository for the MDOT-TruncatedNewton (or MDOT-TNT)
algorithm [1] for solving the entropic-regularized optimal transport (OT) problem. 
In addition to being GPU-friendly, the algorithm is stable under weak regularization and can therefore find highly
precise approximations of the un-regularized problem's solution quickly. 

The current implementation is based on PyTorch and is compatible with both CPU and GPU. PyTorch is the only dependency.


For installation:
First, install PyTorch following the instructions at https://pytorch.org/get-started/locally/ to select the version that matches your system's configuration.
```bash
pip3 install mdot_tnt
```

Quickstart guide:
```
import mdot_tnt
import torch as th
device = 'cuda' if th.cuda.is_available() else 'cpu'
N, M, dim = 100, 200, 128

# Sample row and column marginals from Dirichlet distributions
r = th.distributions.Dirichlet(th.ones(N)).sample()
c = th.distributions.Dirichlet(th.ones(M)).sample()

# Cost matrix from pairwise Euclidean distances squared given random points in R^100
x = th.distributions.MultivariateNormal(th.zeros(dim), th.eye(dim)).sample((N,))
y = th.distributions.MultivariateNormal(th.zeros(dim), th.eye(dim)).sample((M,))
C = th.cdist(x, y, p=2) ** 2
C /= C.max()  # Normalize cost matrix to meet convention.

# Use double precision for numerical stability in high precision regime.
r, c, C = r.double().to(device), c.double().to(device), C.double().to(device)

# Solve OT problem. Increase (decrease) gamma_f for higher (lower) precision.
# Default is gamma_f=2**10. Expect error of order logn / gamma_f at worst, and possibly lower.
cost = mdot_tnt.solve_OT(r, c, C, gamma_f=2**10)

# To return a feasible transport plan, use the following:
transport_plan = mdot_tnt.solve_OT(r, c, C, gamma_f=2**12, return_plan=True)

# In both cases, the default rounding onto the feasible set can be disabled by setting `round=False`.
```

The code is released under a custom non-commerical use license. If you use our work in
your research, please consider citing:

```
@inproceedings{
kemertas2025a,
title={A Truncated Newton Method for Optimal Transport},
author={Mete Kemertas and Amir-massoud Farahmand and Allan Douglas Jepson},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=gWrWUaCbMa}
}
```

For inquiries, email: kemertas [at] cs [dot] toronto [dot] edu

[1] Mete Kemertas, Amir-massoud Farahmand, Allan Douglas Jepson. "A Truncated Newton Method for Optimal Transport." The Thirteenth International Conference on Learning Representations (ICLR), 2025. https://openreview.net/forum?id=gWrWUaCbMa
