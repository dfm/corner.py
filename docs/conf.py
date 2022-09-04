import corner

language = "en"
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]

# General information about the project.
project = "corner.py"
copyright = "2013-2022 Dan Foreman-Mackey"
version = corner.__version__
release = corner.__version__

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "corner.py"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dfm/corner.py",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

nb_execution_mode = "auto"
nb_execution_excludepatterns = []
nb_execution_timeout = -1
