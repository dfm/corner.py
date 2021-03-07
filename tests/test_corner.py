# -*- coding: utf-8 -*-

from collections import OrderedDict

import arviz as az
import numpy as np
import pytest
from matplotlib import pyplot as pl
from matplotlib.testing.decorators import image_comparison

import corner

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import scipy  # noqa
except ImportError:
    scipy_installed = False
else:
    scipy_installed = True


def _run_corner(
    pandas=False,
    arviz=False,
    N=10000,
    seed=1234,
    ndim=3,
    factor=None,
    **kwargs
):
    np.random.seed(seed)
    data1 = np.random.randn(ndim * 4 * N // 5).reshape([4 * N // 5, ndim])
    data2 = 5 * np.random.rand(ndim)[None, :] + np.random.randn(
        ndim * N // 5
    ).reshape([N // 5, ndim])
    data = np.vstack([data1, data2])
    if factor is not None:
        data[:, 0] *= factor
        data[:, 1] /= factor
    if pandas:
        # data = pd.DataFrame.from_items()
        data = pd.DataFrame.from_dict(
            OrderedDict(zip(map("d{0}".format, range(ndim)), data.T))
        )
    elif arviz:
        data = az.from_dict(
            posterior={"x": data[None]},
            sample_stats={"diverging": data[None, :, 0] < 0.0},
        )
        kwargs["truths"] = {"x": np.random.randn(ndim)}

    fig = corner.corner(data, **kwargs)
    return fig


@image_comparison(
    baseline_images=["basic"], remove_text=True, extensions=["png"]
)
def test_basic():
    _run_corner()


@image_comparison(baseline_images=["labels"], extensions=["png"])
def test_labels():
    _run_corner(labels=["a", "b", "c"])


@image_comparison(
    baseline_images=["quantiles"], remove_text=True, extensions=["png"]
)
def test_quantiles():
    _run_corner(quantiles=[0.16, 0.5, 0.84])


@image_comparison(
    baseline_images=["color"], remove_text=True, extensions=["png"]
)
def test_color():
    _run_corner(color="g")


@image_comparison(
    baseline_images=["color_filled"], remove_text=True, extensions=["png"]
)
def test_color_filled():
    _run_corner(color="g", fill_contours=True)


@image_comparison(
    baseline_images=["overplot"], remove_text=True, extensions=["png"]
)
def test_overplot():
    fig = _run_corner(N=15000, color="g", fill_contours=True)
    _run_corner(
        N=5000, factor=0.5, seed=15, color="b", fig=fig, fill_contours=True
    )


@image_comparison(
    baseline_images=["smooth1"], remove_text=True, extensions=["png"]
)
def test_smooth1():
    _run_corner(bins=50)


@pytest.mark.skipif(not scipy_installed, reason="requires scipy for smoothing")
@image_comparison(
    baseline_images=["smooth2"], remove_text=True, extensions=["png"]
)
def test_smooth2():
    _run_corner(bins=50, smooth=1.0)


@pytest.mark.skipif(not scipy_installed, reason="requires scipy for smoothing")
@image_comparison(
    baseline_images=["smooth1d"], remove_text=True, extensions=["png"]
)
def test_smooth1d():
    _run_corner(bins=50, smooth=1.0, smooth1d=1.0)


@image_comparison(baseline_images=["titles1"], extensions=["png"])
def test_titles1():
    _run_corner(show_titles=True)


@image_comparison(baseline_images=["titles2"], extensions=["png"])
def test_titles2():
    _run_corner(show_titles=True, title_fmt=None, labels=["a", "b", "c"])


@image_comparison(
    baseline_images=["top_ticks"], remove_text=True, extensions=["png"]
)
def test_top_ticks():
    _run_corner(top_ticks=True)


@pytest.mark.skipif(pd is None, reason="requires pandas")
@image_comparison(baseline_images=["pandas"], extensions=["png"])
def test_pandas():
    _run_corner(pandas=True)


@image_comparison(
    baseline_images=["truths"], remove_text=True, extensions=["png"]
)
def test_truths():
    _run_corner(truths=[0.0, None, 0.15])


@image_comparison(
    baseline_images=["no_fill_contours"], remove_text=True, extensions=["png"]
)
def test_no_fill_contours():
    _run_corner(no_fill_contours=True)


@image_comparison(
    baseline_images=["tight"], remove_text=True, extensions=["png"]
)
def test_tight():
    _run_corner(ret=True)
    pl.tight_layout()


@image_comparison(
    baseline_images=["reverse"], remove_text=True, extensions=["png"]
)
def test_reverse():
    _run_corner(ndim=2, range=[(4, -4), (-5, 5)])


@image_comparison(
    baseline_images=["hist_bin_factor"], remove_text=True, extensions=["png"]
)
def test_hist_bin_factor():
    _run_corner(hist_bin_factor=4)


@image_comparison(baseline_images=["arviz"], extensions=["png"])
def test_arviz():
    _run_corner(arviz=True)
