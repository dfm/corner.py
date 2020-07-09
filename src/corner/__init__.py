# -*- coding: utf-8 -*-

__all__ = ["corner", "hist2d", "quantile"]

from .corner import corner, hist2d, quantile
from .corner_version import __version__  # noqa

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://corner.readthedocs.io"
__license__ = "BSD"
__description__ = "Make some beautiful corner plots"
__copyright__ = "Copyright 2013-2020 Daniel Foreman-Mackey"
__contributors__ = "https://github.com/dfm/corner.py/graphs/contributors"
__bibtex__ = __citation__ = """
@article{corner,
  doi = {10.21105/joss.00024},
  url = {https://doi.org/10.21105/joss.00024},
  year  = {2016},
  month = {jun},
  publisher = {The Open Journal},
  volume = {1},
  number = {2},
  pages = {24},
  author = {Daniel Foreman-Mackey},
  title = {corner.py: Scatterplot matrices in Python},
  journal = {The Journal of Open Source Software}
}
"""
