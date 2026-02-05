Installation
============

Prerequisites
-------------

MDOT-TNT requires `PyTorch <https://pytorch.org/get-started/locally/>`_.
Install PyTorch for your system configuration first (CPU or CUDA).

Install from PyPI
-----------------

.. code-block:: bash

   pip install mdot-tnt

Install from Source
-------------------

For development or to get the latest version:

.. code-block:: bash

   git clone https://github.com/metekemertas/mdot_tnt.git
   cd mdot_tnt
   pip install -e ".[dev]"

This installs the package in editable mode along with development dependencies
(pytest, ruff, pre-commit, numpy).

Verifying the Installation
--------------------------

.. code-block:: python

   import mdot_tnt
   import torch

   r = torch.rand(10, dtype=torch.float64)
   r = r / r.sum()
   c = torch.rand(12, dtype=torch.float64)
   c = c / c.sum()
   C = torch.rand(10, 12, dtype=torch.float64)

   cost = mdot_tnt.solve_OT(r, c, C, gamma_f=100)
   print(f"Transport cost: {cost:.6f}")
