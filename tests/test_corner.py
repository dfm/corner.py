# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.testing.decorators import image_comparison

import corner


def _run_corner(pandas=False, N=10000, seed=1234, ndim=3, factor=None,
                **kwargs):
    np.random.seed(seed)
    data1 = np.random.randn(ndim*4*N//5).reshape([4*N//5, ndim])
    data2 = (5 * np.random.rand(ndim)[None, :] +
             np.random.randn(ndim*N//5).reshape([N//5, ndim]))
    data = np.vstack([data1, data2])
    if factor is not None:
        data[:, 0] *= factor
        data[:, 1] /= factor
    if pandas:
        data = pd.DataFrame.from_items(zip(map("d{0}".format, range(ndim)),
                                           data.T))

    fig = corner.corner(data, **kwargs)
    return fig


@image_comparison(baseline_images=["basic"], extensions=["png"])
def test_basic():
    _run_corner()


@image_comparison(baseline_images=["labels"], extensions=["png"])
def test_labels():
    _run_corner(labels=["a", "b", "c"])


@image_comparison(baseline_images=["quantiles"], extensions=["png"])
def test_quantiles():
    _run_corner(quantiles=[0.16, 0.5, 0.84])


@image_comparison(baseline_images=["color"], extensions=["png"])
def test_color():
    _run_corner(color="g")


@image_comparison(baseline_images=["color_filled"], extensions=["png"])
def test_color_filled():
    _run_corner(color="g", fill_contours=True)


@image_comparison(baseline_images=["overplot"], extensions=["png"])
def test_overplot():
    fig = _run_corner(color="g", fill_contours=True)
    _run_corner(seed=15, color="b", fig=fig, fill_contours=True)


@image_comparison(baseline_images=["smooth1"], extensions=["png"])
def test_smooth1():
    _run_corner(bins=50)


@image_comparison(baseline_images=["smooth2"], extensions=["png"])
def test_smooth2():
    _run_corner(bins=50, smooth=1.0)


@image_comparison(baseline_images=["smooth1d"], extensions=["png"])
def test_smooth1d():
    _run_corner(bins=50, smooth=1.0, smooth1d=1.0)


@image_comparison(baseline_images=["titles1"], extensions=["png"])
def test_titles1():
    _run_corner(show_titles=True)


@image_comparison(baseline_images=["titles2"], extensions=["png"])
def test_titles2():
    _run_corner(show_titles=True, title_fmt=None, labels=["a", "b", "c"])


@image_comparison(baseline_images=["top_ticks"], extensions=["png"])
def test_top_ticks():
    _run_corner(top_ticks=True)


@image_comparison(baseline_images=["pandas"], extensions=["png"])
def test_pandas():
    _run_corner(pandas=True)


@image_comparison(baseline_images=["truths"], extensions=["png"])
def test_truths():
    _run_corner(truths=[0.0, None, 0.15])


@image_comparison(baseline_images=["no_fill_contours"], extensions=["png"])
def test_no_fill_contours():
    _run_corner(no_fill_contours=True)


@image_comparison(baseline_images=["tight"], extensions=["png"])
def test_tight():
    _run_corner(ret=True)
    pl.tight_layout()


@image_comparison(baseline_images=["reverse"], extensions=["png"])
def test_reverse():
    _run_corner(ndim=2, range=[(4, -4), (-5, 5)])


@image_comparison(baseline_images=["hist_bin_factor"], extensions=["png"])
def test_hist_bin_factor():
    _run_corner(hist_bin_factor=4)
