# -*- coding: utf-8 -*-

import corner

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]
master_doc = "index"

# General information about the project.
project = "corner.py"
copyright = "2013-2021 Dan Foreman-Mackey & contributors"

version = corner.__version__
release = corner.__version__

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "corner.py"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dfm/corner.py",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://corner.readthedocs.io/en/latest/"
nb_execution_mode = "force"

# download notebooks as .ipynb and not as .ipynb.txt
html_sourcelink_suffix = ""
