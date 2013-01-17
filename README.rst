triangle.py
===========

Make some beautiful corner plots.

Corner plot /ˈkôrnər plät/ (noun):
    An illustrative representation of different projections of samples in
    high dimensional spaces. It is awesome. I promise.

Built by `Dan Foreman-Mackey <http://dan.iel.fm>`_ and collaborators (see
``triangle.__contributors__`` for the most up to date list). Licensed under
the 2-clause BSD license (see ``LICENSE``).


Installation
------------

Just run

::

    pip install triangle_plot

to get the most recent stable version.


Usage
-----

The main entry point is the ``triangle.corner`` function. You'll just use it
like this:

::

    import numpy as np
    import triangle

    ndim, nsamples = 5, 10000
    samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
    figure = triangle.corner(samples)
    figure.savefig("triangle.png")
