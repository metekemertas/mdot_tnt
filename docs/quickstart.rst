Quick Start
===========

Single Problem
--------------

.. code-block:: python

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

Batched Solving
---------------

When solving multiple OT problems, use the batched solver for significant speedup
compared to sequential solution:

.. code-block:: python

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

The batched solver achieves speedup by amortizing GPU synchronization overhead
across all problems in the batch.

Performance Tips
----------------

1. **Use float64** for ``gamma_f > 1024`` (automatic conversion with warning).
2. **Normalize cost matrices** to [0, 1] for numerical stability.
3. **Use batched solver** when solving multiple problems with shared structure.
4. **Increase** ``gamma_f`` for higher precision (error scales as
   :math:`O(\log n / \gamma)` in the worst case, but can be much better).
