#!/usr/bin/env python


import numpy as np
from matplotlib import rcParams

import corner

rcParams["font.size"] = 16
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

np.random.seed(42)

# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.
data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape(
    [4 * nsamples / 5, ndim]
)
data2 = 4 * np.random.rand(ndim)[None, :] + np.random.randn(
    ndim * nsamples / 5
).reshape([nsamples / 5, ndim])
data = np.vstack([data1, data2])

# Plot it.
figure = corner.corner(
    data,
    labels=[
        r"$x$",
        r"$y$",
        r"$\log \alpha$",
        r"$\Gamma \, [\mathrm{parsec}]$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
figure.gca().annotate(
    "a demo corner plot",
    xy=(1.0, 1.0),
    xycoords="figure fraction",
    xytext=(-20, -10),
    textcoords="offset points",
    ha="right",
    va="top",
)
figure.savefig("demo.png", dpi=300)
