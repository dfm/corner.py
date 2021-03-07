# corner.py

_Make some beautiful corner plots._

```{admonition} Corner plot /ˈkôrnər plät/ (noun):
:class: tip

An illustrative representation of different projections of samples in
high dimensional spaces. It is awesome. I promise.
_Synonyms: scatterplot matrix, pairs plot, draftsman's display_
```

This Python module uses [matplotlib](https://matplotlib.org/) to visualize
multidimensional samples using a scatterplot matrix.
In these visualizations, each one- and two-dimensional projection of the
sample is plotted to reveal covariances.
_corner_ was originally conceived to display the results of Markov Chain
Monte Carlo simulations and the defaults are chosen with this application in
mind but it can be used for displaying many qualitatively different samples.

Development of _corner_ happens [on GitHub](https://github.com/dfm/corner.py) so you can [raise any issues you have there](https://github.com/dfm/corner.py/issues).
_corner_ has been used extensively in the astronomical literature and it [has occasionally been cited](https://ui.adsabs.harvard.edu/abs/2016JOSS....1...24F/citations)
as `corner.py` or using its previous name `triangle.py`.

[![Tests](https://github.com/dfm/corner.py/workflows/Tests/badge.svg?style=flat)](https://github.com/dfm/corner.py/actions)
[![License](https://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/dfm/corner.py/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/4729/dfm/corner.py.svg?style=flat)](https://zenodo.org/badge/latestdoi/4729/dfm/corner.py)
[![Paper](https://joss.theoj.org/papers/10.21105/joss.00024/status.svg?style=flat)](http://dx.doi.org/10.21105/joss.00024)

## Documentation

```{toctree}
:maxdepth: 2

install
pages/quickstart
pages/sigmas
pages/custom
pages/arviz
api
```

## Attribution

If you make use of this code, please cite [the JOSS paper](https://dx.doi.org/10.21105/joss.00024):

```tex
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
```

## Authors & License

Copyright 2013-2021 Dan Foreman-Mackey

Built by [Dan Foreman-Mackey](https://github.com/dfm) and contributors (see
[the contribution graph](https://github.com/dfm/corner.py/graphs/contributors) for the most
up to date list). Licensed under the 2-clause BSD license (see `LICENSE`).
