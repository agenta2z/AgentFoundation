# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CFR Models Documentation"
copyright = "2024, Meta"
author = "Meta CFR Team"
release = "1.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Try Read the Docs theme, fall back to alabaster if not available
try:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"
html_static_path = ["_static"]

# Theme options for Read the Docs theme
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Custom sidebar templates
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "searchbox.html",
    ]
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Pygments (syntax highlighting) style ------------------------------------
pygments_style = "sphinx"

# -- Additional options ------------------------------------------------------
# Suppress warnings about missing references to external documents
suppress_warnings = ["ref.option"]

# -- Options for code blocks -------------------------------------------------
highlight_language = "python"
