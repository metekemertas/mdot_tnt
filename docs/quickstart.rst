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

Low-Memory / Point Cloud Solving
---------------------------------

For large problems where allocating a full ``n × m`` cost matrix is prohibitive,
use :func:`mdot_tnt.lowmem.solve_OT_lowmem`.  It processes columns in blocks of
size ``block_size`` (k), keeping only O(nk) entries in working memory at a time.

**Point cloud mode** (true O(nk + (n+m)d) memory — C is never materialised):

.. code-block:: python

   import torch
   from mdot_tnt.lowmem import solve_OT_lowmem

   device = "cuda" if torch.cuda.is_available() else "cpu"

   n, m, d = 10000, 10000, 64
   X = torch.rand(n, d, device=device, dtype=torch.float64)
   Y = torch.rand(m, d, device=device, dtype=torch.float64)
   r = torch.ones(n, device=device, dtype=torch.float64) / n
   c = torch.ones(m, device=device, dtype=torch.float64) / m

   # block_size controls the memory / speed trade-off
   cost = solve_OT_lowmem(r, c, X=X, Y=Y, gamma_f=1024, block_size=512)

**Dense matrix mode** (full C provided, but only k columns loaded at once):

.. code-block:: python

   C = torch.rand(n, m, device=device, dtype=torch.float64)
   cost = solve_OT_lowmem(r, c, C=C, gamma_f=1024, block_size=512)

Memory / speed trade-off via ``block_size``:

- ``block_size = m`` (default): fastest, same memory as :func:`~mdot_tnt.solve_OT`
- ``block_size = int(m**0.5)``: good balance
- ``block_size = 1``: minimum memory (slowest)

Performance Tips
----------------

1. **Use float64** for ``gamma_f > 1024`` (automatic conversion with warning).
2. **Normalize cost matrices** to [0, 1] for numerical stability.
3. **Use batched solver** when solving multiple problems with shared structure.
4. **Increase** ``gamma_f`` for higher precision (error scales as
   :math:`O(\log n / \gamma)` in the worst case, but can be much better).
