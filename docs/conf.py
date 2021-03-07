# -*- coding: utf-8 -*-

import os

import corner

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = u"corner.py"
copyright = u"2013-2021 Dan Foreman-Mackey & contributors"

version = corner.__version__
release = corner.__version__

exclude_patterns = ["_build", "_static/notebooks/profile"]
pygments_style = "sphinx"

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]
html_show_sourcelink = False

nbsphinx_execute = "always"
nbsphinx_prolog = """
.. note:: This page was generated from an IPython notebook that can be downloaded
          `here <https://github.com/dfm/corner.py/blob/main/docs/{{ env.doc2path(env.docname, base=None) }}>`_.

"""
