# MDOT-TNT

<img src="assets/logo.png" alt="MDOT-TNT Logo" width="180" align="right"/>

**A Truncated Newton Method for Optimal Transport**

[![PyPI version](https://badge.fury.io/py/mdot-tnt.svg)](https://badge.fury.io/py/mdot-tnt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-02B36C)](LICENSE)

A fast, GPU-accelerated solver for entropic-regularized optimal transport (OT) problems. MDOT-TNT combines mirror descent with a truncated Newton projection method to achieve high numerical precision while remaining stable under weak regularization.

<br clear="right"/>

## Features

- **High Precision**: Stable under extremely weak regularization  (γ up to 2¹⁸), enabling highly precise approximations of unregularized OT
- **GPU Accelerated**: Fully compatible with CUDA for fast computation on large problems
- **Batched Solving**: Solve multiple OT problems simultaneously in batched mode
- **Memory Efficient**: Log-domain computations and efficient rounding avoid storing full transport plans
- **PyTorch Native**: Seamless integration with PyTorch, supporting autograd-compatible inputs

## Installation

**Prerequisites**: Install [PyTorch](https://pytorch.org/get-started/locally/) for your system configuration first.

```bash
pip install mdot-tnt
```

For development:

```bash
git clone https://github.com/metekemertas/mdot_tnt.git
cd mdot_tnt
pip install -e ".[dev]"
```

## Quick Start

### Single Problem

```python
import torch
import mdot_tnt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create marginals (probability distributions)
n, m = 512, 512
r = torch.rand(n, device=device, dtype=torch.float64)
r = r / r.sum()
c = torch.rand(m, device=device, dtype=torch.float64)
c = c / c.sum()

# Cost matrix (e.g., pairwise distances)
C = torch.rand(n, m, device=device, dtype=torch.float64)

# Solve for optimal transport cost
cost = mdot_tnt.solve_OT(r, c, C, gamma_f=1024)

# Or get the full transport plan
plan = mdot_tnt.solve_OT(r, c, C, gamma_f=1024, return_plan=True)
```

### Batched Solving

When solving multiple OT problems, use the batched solver for significant speedup compared to sequential solution:

```python
import torch
import mdot_tnt

device = "cuda"
batch_size, n, m = 32, 512, 512

# Multiple marginal pairs
r = torch.rand(batch_size, n, device=device, dtype=torch.float64)
r = r / r.sum(-1, keepdim=True)
c = torch.rand(batch_size, m, device=device, dtype=torch.float64)
c = c / c.sum(-1, keepdim=True)

# Shared cost matrix (or per-problem: shape [batch_size, n, m])
C = torch.rand(n, m, device=device, dtype=torch.float64)

# Solve all problems at once
costs = mdot_tnt.solve_OT_batched(r, c, C, gamma_f=1024)  # Returns (batch_size,) tensor
```

The batched solver achieves speedup by amortizing GPU synchronization overhead across all problems in the batch.

## API Reference

### `solve_OT`

```python
mdot_tnt.solve_OT(r, c, C, gamma_f=1024., return_plan=False, round=True, log=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | `Tensor` | Row marginal of shape `(n,)`, must sum to 1 |
| `c` | `Tensor` | Column marginal of shape `(m,)`, must sum to 1 |
| `C` | `Tensor` | Cost matrix of shape `(n, m)`, recommended to normalize to [0, 1] |
| `gamma_f` | `float` | Temperature parameter (inverse regularization). Higher = more accurate. Default: 1024 |
| `return_plan` | `bool` | If True, return transport plan instead of cost |
| `round` | `bool` | If True, round solution onto feasible set |
| `log` | `bool` | If True, also return optimization logs |

**Returns**: Transport cost (scalar) or plan `(n, m)`, optionally with logs dict.

### `solve_OT_batched`

```python
mdot_tnt.solve_OT_batched(r, c, C, gamma_f=1024., return_plan=False, round=True, log=False)
```

Same parameters as `solve_OT`, but with batched inputs:
- `r`: Shape `(batch, n)`
- `c`: Shape `(batch, m)`  
- `C`: Shape `(n, m)` for shared cost, or `(batch, n, m)` for per-problem costs

**Returns**: Costs `(batch,)` or plans `(batch, n, m)`.

## Performance Tips

1. **Use float64** for `gamma_f > 1024` (automatic conversion with warning)
2. **Normalize cost matrices** to [0, 1] for numerical stability
3. **Use batched solver** when solving multiple problems with shared structure
4. **Increase `gamma_f`** for higher precision (error scales as O(log n / γ) in the worst case, but can be much better)

## Citation

If you use MDOT-TNT in your research, please cite:

```bibtex
@inproceedings{kemertas2025truncated,
  title={A Truncated Newton Method for Optimal Transport},
  author={Kemertas, Mete and Farahmand, Amir-massoud and Jepson, Allan Douglas},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=gWrWUaCbMa}
}
```

## License

This code is released under the [BSD 3-Clause license.](LICENSE).

## Contact

For questions or issues, please [open an issue](https://github.com/metekemertas/mdot_tnt/issues) or email: kemertas [at] cs [dot] toronto [dot] edu
