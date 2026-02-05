"""Sphinx configuration for MDOT-TNT documentation."""

import os
import shutil
import sys

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# Copy tutorial notebooks into docs/tutorial/ so Sphinx can find them
# without relying on symlinks (which can be fragile across platforms).
_tutorial_src = os.path.join(os.path.abspath(".."), "tutorial")
_tutorial_dst = os.path.join(os.path.abspath("."), "tutorial")
if os.path.isdir(_tutorial_src):
    os.makedirs(_tutorial_dst, exist_ok=True)
    for _nb in os.listdir(_tutorial_src):
        if _nb.endswith(".ipynb"):
            shutil.copy2(os.path.join(_tutorial_src, _nb), _tutorial_dst)

# -- Project information -----------------------------------------------------

project = "MDOT-TNT"
copyright = "2025, Mete Kemertas"
author = "Mete Kemertas"
release = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Options for Napoleon (Google-style docstrings) --------------------------

napoleon_google_docstrings = True
napoleon_numpy_docstrings = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for nbsphinx ----------------------------------------------------

nbsphinx_execute = "never"  # Don't re-execute notebooks during build

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = ["_static"]
html_logo = "../assets/logo.png"
html_favicon = "../assets/logo.png"
