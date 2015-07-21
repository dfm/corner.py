# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_hist2d", "test_corner"]

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

import triangle

FIGURE_PATH = "test_figures"


def _run_hist2d(nm, N=50000, seed=1234, **kwargs):
    print(" .. {0}".format(nm))

    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    # Generate some fake data.
    np.random.seed(seed)
    x = np.random.randn(N)
    y = np.random.randn(N)

    fig, ax = pl.subplots(1, 1, figsize=(8, 8))
    triangle.hist2d(x, y, ax=ax, **kwargs)
    fig.savefig(os.path.join(FIGURE_PATH, "hist2d_{0}.png".format(nm)))
    pl.close(fig)


def test_hist2d():
    _run_hist2d("cutoff", range=[(0, 4), (0, 2.5)])
    _run_hist2d("cutoff2", range=[(-4, 4), (-0.1, 0.1)], N=100000,
                fill_contours=True, smooth=1)
    _run_hist2d("basic")
    _run_hist2d("color", color="g")
    _run_hist2d("levels1", levels=[0.68, 0.95])
    _run_hist2d("levels2", levels=[0.5, 0.75])
    _run_hist2d("filled", fill_contours=True)
    _run_hist2d("smooth1", bins=50)
    _run_hist2d("smooth2", bins=50, smooth=(1.0, 1.5))
    _run_hist2d("philsplot", plot_datapoints=False, fill_contours=True,
                levels=[0.68, 0.95], color="g", bins=50, smooth=1.)


def _run_corner(nm, pandas=False, N=10000, seed=1234, ndim=3, ret=False,
                **kwargs):
    print(" .. {0}".format(nm))

    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    np.random.seed(seed)
    data1 = np.random.randn(ndim*4*N/5.).reshape([4*N/5., ndim])
    data2 = (5 * np.random.rand(ndim)[None, :]
             + np.random.randn(ndim*N/5.).reshape([N/5., ndim]))
    data = np.vstack([data1, data2])
    if pandas:
        data = pd.DataFrame.from_items(zip(map("d{0}".format, range(ndim)),
                                           data.T))

    fig = triangle.corner(data, **kwargs)
    fig.savefig(os.path.join(FIGURE_PATH, "triangle_{0}.png".format(nm)))
    if ret:
        return fig
    else:
        pl.close(fig)


def test_corner():
    _run_corner("basic")
    _run_corner("labels", labels=["a", "b", "c"])
    _run_corner("quantiles", quantiles=[0.16, 0.5, 0.84])
    _run_corner("color", color="g")
    fig = _run_corner("color-filled", color="g", fill_contours=True,
                      ret=True)
    _run_corner("overplot", seed=15, color="b", fig=fig, fill_contours=True)
    _run_corner("smooth1", bins=50)
    _run_corner("smooth2", bins=50, smooth=1.0)
    _run_corner("smooth1d", bins=50, smooth=1.0, smooth1d=1.0)
    _run_corner("titles1", show_titles=True)
    _run_corner("top-ticks", top_ticks=True)
    _run_corner("pandas", pandas=True)
    _run_corner("truths", truths=[0.0, None, 0.15])


if __name__ == "__main__":
    print("Testing 'hist2d'")
    test_hist2d()

    print("Testing 'corner'")
    test_corner()
