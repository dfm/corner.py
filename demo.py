#!/usr/bin/env python

import numpy as np
import triangle

# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.
data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape([4 * nsamples / 5,
                                                          ndim])
data2 = (5 * np.random.rand(ndim)[None, :]
         + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
data = np.vstack([data1, data2])

# Plot it.
figure = triangle.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$",
                                       r"$\Gamma \, [\mathrm{parsec}]$"],
                         quantiles=[0.16, 0.5, 0.84])
figure.savefig("demo.png")
