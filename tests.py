# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import numpy as np
import matplotlib.pyplot as pl

import triangle

FIGURE_PATH = "test_figures"


def _run_hist2d(nm, N=50000, **kwargs):
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    # Generate some fake data.
    x = np.random.randn(N)
    y = np.random.randn(N)

    fig, ax = pl.subplots(1, 1, figsize=(8, 8))
    triangle.hist2d(x, y, ax=ax, **kwargs)
    fig.savefig(os.path.join(FIGURE_PATH, "hist2d_{0}.png".format(nm)))
    pl.close(fig)


if __name__ == "__main__":
    _run_hist2d("basic")
    _run_hist2d("color", color="g")
    _run_hist2d("levels1", levels=[0.68, 0.95])
    _run_hist2d("levels2", levels=[0.5, 0.75])
    _run_hist2d("filled", fill_contours=True)
