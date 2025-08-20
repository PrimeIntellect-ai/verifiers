# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root to Python path for importing verifiers
sys.path.insert(0, os.path.abspath("../.."))  # ← up two levels to project root

# Import verifiers to get version dynamically
import verifiers

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "verifiers"
copyright = "2025, William Brown"
author = "William Brown"
release = verifiers.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "tasklist",
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Verifiers"
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}
# html_static_path = ["_static"]
