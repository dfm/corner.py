.. _install:

Installation
============

Dependencies
------------

corner.py depends on ``matplotlib``, ``numpy``, and optionally ``scipy``. You
can install these using your favorite Python package manager and I would
recommend `conda <http://conda.pydata.org/docs/>`_ if you don't already have
an opinion.

Using pip
---------

The easiest way to install the most recent stable version of ``corner`` is
with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    pip install corner


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/dfm/corner.py/tarball/master>`_ or cloning `the git
repository <https://github.com/dfm/corner.py>`_:

.. code-block:: bash

    git clone https://github.com/dfm/corner.py.git

Once you've downloaded the source, you can navigate into the root source
directory and run:

.. code-block:: bash

    python setup.py install


Tests
-----

If you installed from source, you can run the unit tests using `nose
<http://nose.readthedocs.org/>`_. From the root of the source directory, run:

.. code-block:: bash

    nosetests corner

This might take a few minutes but you shouldn't get any errors if all went
as planned.
