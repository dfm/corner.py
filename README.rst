corner.py
=========

Make some beautiful corner plots.

Corner plot /ˈkôrnər plät/ (noun):
    An illustrative representation of different projections of samples in
    high dimensional spaces. It is awesome. I promise.

Built by `Dan Foreman-Mackey <http://dan.iel.fm>`_ and collaborators (see
``corner.__contributors__`` for the most up to date list). Licensed under
the 2-clause BSD license (see ``LICENSE``).


Installation
------------

Just run

::

    pip install corner

to get the most recent stable version.


Usage
-----

The main entry point is the ``corner.corner`` function. You'll just use it
like this:

::

    import numpy as np
    import corner

    ndim, nsamples = 5, 10000
    samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
    figure = corner.corner(samples)
    figure.savefig("corner.png")

With some other tweaks (see `demo.py
<https://github.com/dfm/corner.py/blob/master/demo.py>`_) you can get
something that looks awesome like:

.. image:: https://raw.github.com/dfm/corner.py/master/corner.png

By default, data points are shown as grayscale points with contours.
Contours are shown at 0.5, 1, 1.5, and 2 sigma.

For more usage examples, take a look at `tests.py
<https://github.com/dfm/corner.py/blob/master/tests.py>`_.


Documentation
-------------

All the options are documented in the docstrings for the ``corner`` and
``hist2d`` functions. These can be viewed in a Python shell using:

::

    import corner
    print(corner.corner.__doc__)

or, in IPython using:

::

    import corner
    corner.corner?


A note about "sigmas"
+++++++++++++++++++++

We are regularly asked about the "sigma" levels in the 2D histograms. These
are not the 68%, *etc.* values that we're used to for 1D distributions. In two
dimensions, a Gaussian density is given by:

::

    pdf(r) = exp(-(r/s)^2/2) / (2*pi*s^2)

The integral under this density is:

::

    cdf(x) = Integral(r * exp(-(r/s)^2/2) / s^2, {r, 0, x})
           = 1 - exp(-(x/s)^2/2)

This means that within "1-sigma", the Gaussian contains ``1-exp(-0.5) ~ 0.393``
or 39.3% of the volume. Therefore the relevant 1-sigma levels for a 2D
histogram of samples is 39% not 68%. If you must use 68% of the mass, use the
``levels`` keyword argument.

The `"sigma-demo" notebook
<https://github.com/dfm/corner.py/blob/master/sigma-demo.ipynb>`_ visually
demonstrates the difference between these choices of levels.


Attribution
-----------

.. image:: https://zenodo.org/badge/4729/dfm/corner.py.svg
   :target: https://zenodo.org/badge/latestdoi/4729/dfm/corner.py

If you make use of this code, please `cite it
<https://zenodo.org/badge/latestdoi/4729/dfm/corner.py>`_.


License
-------

Copyright 2013, 2014 Dan Foreman-Mackey

corner.py is free software made available under the BSD License.
For details see the LICENSE file.
