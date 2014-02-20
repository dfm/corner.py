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
test_data = np.vstack([data1, data2])

# Generate some more fake data.
data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape([4 * nsamples / 5,
                                                          ndim])
data2 = (5 * np.random.rand(ndim)[None, :]
         + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
test_data2 = np.vstack([data1, data2])

# Plot it.
fig = triangle.corner(test_data, labels=[r"$x$", r"$y$", r"$z$"],
                      truths=[0.5, 0.5, 0.45],
                      quantiles=[0.16, 0.5, 0.84],
                      plot_contours=True, plot_datapoints=False,
                      contour_kwargs=dict(linewidths=3.,alpha=0.5),
                      hist_kwargs=dict(normed=True,bins=50))

fig = triangle.corner(test_data2, fig=fig,
                      quantiles=[0.16, 0.5, 0.84],
                      plot_contours=False, plot_datapoints=True,
                      point_kwargs=dict(color='#2b8cbe',markersize=4,alpha=0.02),
                      hist_kwargs=dict(color='#2b8cbe',bins=100,normed=True))

fig.gca().annotate("A Title", xy=(0.5, 1.0), xycoords="figure fraction",
                      xytext=(0, -5), textcoords="offset points",
                      ha="center", va="top")
fig.savefig("demo.png")
