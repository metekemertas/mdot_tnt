MDOT-TNT
========

**A Truncated Newton Method for Optimal Transport**

.. image:: ../assets/logo.png
   :width: 180
   :align: right
   :alt: MDOT-TNT Logo

MDOT-TNT is a fast, GPU-accelerated solver for entropic-regularized optimal transport (OT)
problems. It combines mirror descent with a truncated Newton projection method to achieve
high numerical precision while remaining stable under weak regularization.

Key features:

- **High Precision**: Stable under extremely weak regularization (γ up to 2\ :sup:`18`),
  enabling highly precise approximations of unregularized OT
- **GPU Accelerated**: Fully compatible with CUDA for fast computation on large problems
- **Batched Solving**: Solve multiple OT problems simultaneously in batched mode
- **Memory Efficient**: Log-domain computations and efficient rounding avoid storing full
  transport plans
- **PyTorch Native**: Seamless integration with PyTorch, supporting autograd-compatible inputs

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api
   tutorial

Citation
--------

If you use MDOT-TNT in your research, please cite:

.. code-block:: bibtex

   @inproceedings{kemertas2025truncated,
     title={A Truncated Newton Method for Optimal Transport},
     author={Kemertas, Mete and Farahmand, Amir-massoud and Jepson, Allan Douglas},
     booktitle={The Thirteenth International Conference on Learning Representations},
     year={2025},
     url={https://openreview.net/forum?id=gWrWUaCbMa}
   }

License
-------

This code is released under a non-commercial use license. For commercial licensing
inquiries, please contact the authors.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
