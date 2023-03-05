# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as pl
from matplotlib.testing.decorators import image_comparison

import corner

try:
    import scipy  # noqa
except ImportError:
    scipy_installed = False
else:
    scipy_installed = True


def _run_hist2d(nm, N=50000, seed=1234, **kwargs):
    # Generate some fake data.
    np.random.seed(seed)
    x = np.random.randn(N)
    y = np.random.randn(N)

    fig, ax = pl.subplots(1, 1, figsize=(8, 8))
    corner.hist2d(x, y, ax=ax, **kwargs)


@image_comparison(
    baseline_images=["cutoff"], remove_text=True, extensions=["png"]
)
def test_cutoff():
    _run_hist2d("cutoff", range=[(0, 4), (0, 2.5)])


@pytest.mark.skipif(not scipy_installed, reason="requires scipy for smoothing")
@image_comparison(
    baseline_images=["cutoff2"], remove_text=True, extensions=["png"]
)
def test_cutoff2():
    _run_hist2d(
        "cutoff2",
        range=[(-4, 4), (-0.1, 0.1)],
        N=100000,
        fill_contours=True,
        smooth=1,
    )


@image_comparison(
    baseline_images=["basic"], remove_text=True, extensions=["png"]
)
def test_basic():
    _run_hist2d("basic")


@image_comparison(
    baseline_images=["color"], remove_text=True, extensions=["png"]
)
def test_color():
    _run_hist2d("color", color="g")


@image_comparison(
    baseline_images=["backgroundDark"], remove_text=True, extensions=["png"]
)
def test_backgroundDark():
    pl.style.use("dark_background")
    _run_hist2d("backgroundDark")
    pl.style.use("default")


@image_comparison(
    baseline_images=["backgroundDark2"], remove_text=True, extensions=["png"]
)
def test_backgroundDark2():
    pl.style.use("dark_background")
    _run_hist2d("backgroundDark2", color="r")
    pl.style.use("default")


@image_comparison(
    baseline_images=["backgroundSolarized"],
    remove_text=True,
    extensions=["png"],
)
def test_backgroundSolarized():
    pl.style.use("Solarize_Light2")
    _run_hist2d("backgroundSolarized")
    pl.style.use("default")


@image_comparison(
    baseline_images=["backgroundColor"], remove_text=True, extensions=["png"]
)
def test_backgroundColor():
    pl.style.use("default")
    matplotlib.rcParams["axes.facecolor"] = "yellow"
    matplotlib.rcParams["axes.edgecolor"] = "red"
    matplotlib.rcParams["xtick.color"] = "green"
    matplotlib.rcParams["ytick.color"] = "blue"
    _run_hist2d("backgroundColor")
    pl.style.use("default")


@image_comparison(
    baseline_images=["levels1"], remove_text=True, extensions=["png"]
)
def test_levels1():
    _run_hist2d("levels1", levels=[0.68, 0.95])


@image_comparison(
    baseline_images=["levels2"], remove_text=True, extensions=["png"]
)
def test_levels2():
    _run_hist2d("levels2", levels=[0.5, 0.75])


@image_comparison(
    baseline_images=["filled"], remove_text=True, extensions=["png"]
)
def test_filled():
    _run_hist2d("filled", fill_contours=True)


@image_comparison(
    baseline_images=["smooth1"], remove_text=True, extensions=["png"]
)
def test_smooth1():
    _run_hist2d("smooth1", bins=50)


@pytest.mark.skipif(not scipy_installed, reason="requires scipy for smoothing")
@image_comparison(
    baseline_images=["smooth2"], remove_text=True, extensions=["png"]
)
def test_smooth2():
    _run_hist2d("smooth2", bins=50, smooth=(1.0, 1.5))


@pytest.mark.skipif(not scipy_installed, reason="requires scipy for smoothing")
@image_comparison(
    baseline_images=["philsplot"], remove_text=True, extensions=["png"]
)
def test_philsplot():
    _run_hist2d(
        "philsplot",
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.68, 0.95],
        color="g",
        bins=50,
        smooth=1.0,
    )


@image_comparison(
    baseline_images=["lowN"], remove_text=True, extensions=["png"]
)
def test_lowN():
    _run_hist2d("lowN", N=20)


@image_comparison(
    baseline_images=["lowNfilled"], remove_text=True, extensions=["png"]
)
def test_lowNfilled():
    _run_hist2d("lowNfilled", N=20, fill_contours=True)


@image_comparison(
    baseline_images=["lowNnofill"], remove_text=True, extensions=["png"]
)
def test_lowNnofill():
    _run_hist2d("lowNnofill", N=20, no_fill_contours=True)


def test_infinite_loop():
    x, y = np.random.rand(2, 1000)
    with pytest.raises(ValueError):
        corner.hist2d(x, y, 20, range=[(0, 1), (2, 3)])
