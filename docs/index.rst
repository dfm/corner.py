
corner.py
=========

*Make some beautiful corner plots.*

Corner plot /ˈkôrnər plät/ (noun):
    An illustrative representation of different projections of samples in
    high dimensional spaces. It is awesome. I promise.

    *Synonyms: scatterplot matrix, pairs plot, draftsman's display*


This Python module uses `matplotlib <http://matplotlib.org/>`_ to visualize
multidimensional samples using a scatterplot matrix.
In these visualizations, each one- and two-dimensional projection of the
sample is plotted to reveal covariances.
*corner* was originally conceived to display the results of Markov Chain
Monte Carlo simulations and the defaults are chosen with this application in
mind but it can be used for displaying many qualitatively different samples.

Development of *corner* happens `on GitHub
<https://github.com/dfm/corner.py>`_ so you can `raise any issues you have
there <https://github.com/dfm/corner.py/issues>`_.
*corner* has been used extensively in the astronomical literature and it `has
occasionally been cited
<https://ui.adsabs.harvard.edu/#search/q=%22triangle.py%22%20or%20%22corner.py%22&sort=date%20desc>`_
as ``corner.py`` or using its previous name ``triangle.py``.



.. image:: http://img.shields.io/travis/dfm/corner.py/master.svg?style=flat
    :target: https://travis-ci.org/dfm/corner.py
.. image:: https://coveralls.io/repos/github/dfm/corner.py/badge.svg?branch=master&style=flat
    :target: https://coveralls.io/github/dfm/corner.py?branch=master&style=flat
.. image:: http://img.shields.io/badge/license-BSD-blue.svg?style=flat
    :target: https://github.com/dfm/corner.py/blob/master/LICENSE
.. image:: https://zenodo.org/badge/4729/dfm/corner.py.svg?style=flat
    :target: https://zenodo.org/badge/latestdoi/4729/dfm/corner.py


Documentation
-------------

.. toctree::
   :maxdepth: 2

   install
   pages/quickstart
   pages/sigmas
   pages/custom
   api



Attribution
-----------

If you make use of this code, please cite `the JOSS paper
<http://dx.doi.org/10.21105/joss.00024>`_:

.. code-block:: tex

    @article{corner,
        Author = {Daniel Foreman-Mackey},
        Doi = {10.21105/joss.00024},
        Title = {corner.py: Scatterplot matrices in Python},
        Journal = {The Journal of Open Source Software},
        Year = 2016,
        Volume = 24,
        Url = {http://dx.doi.org/10.5281/zenodo.45906}
    }


Authors & License
-----------------

Copyright 2013-2016 Dan Foreman-Mackey & contributors

Built by `Dan Foreman-Mackey <https://github.com/dfm>`_ and contributors (see
``corner.__contributors__`` for the most up to date list). Licensed under
the 2-clause BSD license (see ``LICENSE``).
