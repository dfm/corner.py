#!/usr/bin/env python

import numpy as np

print "Importing triangle module..."
import triangle

# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.

print "Generating",nsamples,"samples in",ndim,"dimensions..."
data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape([4 * nsamples / 5,
                                                          ndim])
data2 = (5 * np.random.rand(ndim)[None, :]
         + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
data = np.vstack([data1, data2])

# Plot it.
print "Making cornerplot, with 68 and 95% credible regions..."
figure = triangle.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$",
                                       r"$\Gamma \, [\mathrm{parsec}]$"],
                         truths=[0.0, 0.0, 0.0], truth_color='red',
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 12},
                         plot_datapoints=True, fill_contours=True,
                         levels=[0.68, 0.95], color="k", bins=100, smooth=0.1)

figure.gca().annotate("Visualizing multivariate PDFs with triangle.py",
                      xy=(0.5, 1.0), xycoords="figure fraction",
                      xytext=(0, -5), textcoords="offset points",
                      ha="center", va="top")

print "Saving to file..."
figure.savefig("demo.png")

print "Check out the plot in demo.png "
