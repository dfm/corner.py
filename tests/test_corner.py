# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import pytest
from matplotlib import pyplot as pl
from matplotlib.testing.decorators import image_comparison

import corner


def _run_corner(
    pandas=False,
    arviz=False,
    N=10000,
    seed=1234,
    ndim=3,
    factor=None,
    exp_data=False,
    **kwargs,
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
    if exp_data:
        data = 10**data
    if pandas:
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame.from_dict(
            OrderedDict(zip(map("d{0}".format, range(ndim)), data.T))
        )
    elif arviz:
        az = pytest.importorskip("arviz")
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


@image_comparison(
    baseline_images=["basic_log"], remove_text=True, extensions=["png"]
)
def test_basic_log():
    _run_corner(exp_data=True, axes_scale="log")


@image_comparison(
    baseline_images=["basic_log_x2_only"], remove_text=True, extensions=["png"]
)
def test_basic_log_x2_only():
    _run_corner(exp_data=True, axes_scale=["linear", "log", "linear"])


@image_comparison(baseline_images=["labels"], extensions=["png"])
def test_labels():
    _run_corner(labels=["a", "b", "c"])


@image_comparison(
    baseline_images=["quantiles"], remove_text=True, extensions=["png"]
)
def test_quantiles():
    _run_corner(quantiles=[0.16, 0.5, 0.84])


@image_comparison(
    baseline_images=["quantiles_log"], remove_text=True, extensions=["png"]
)
def test_quantiles_log():
    _run_corner(exp_data=True, axes_scale="log", quantiles=[0.16, 0.5, 0.84])


@image_comparison(
    baseline_images=["title_quantiles"], remove_text=False, extensions=["png"]
)
def test_title_quantiles():
    _run_corner(
        quantiles=[0.16, 0.5, 0.84],
        title_quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
    )


@image_comparison(
    baseline_images=["title_quantiles_default"],
    remove_text=False,
    extensions=["png"],
)
def test_title_quantiles_default():
    _run_corner(quantiles=[0.16, 0.5, 0.84], show_titles=True)


@image_comparison(
    baseline_images=["title_quantiles_raises"],
    remove_text=False,
    extensions=["png"],
)
def test_title_quantiles_raises():
    with pytest.raises(ValueError):
        _run_corner(quantiles=[0.05, 0.16, 0.5, 0.84, 0.95], show_titles=True)

    # This one shouldn't raise since show_titles isn't provided
    _run_corner(quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])


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
    baseline_images=["overplot_1d"], remove_text=True, extensions=["png"]
)
def test_overplot_1d():
    fig = _run_corner(N=15000, ndim=1, color="g", fill_contours=True)
    _run_corner(
        N=5000, ndim=1, seed=15, color="b", fig=fig, fill_contours=True
    )


@image_comparison(
    baseline_images=["overplot_log"], remove_text=True, extensions=["png"]
)
def test_overplot_log():
    fig = _run_corner(
        N=15000,
        exp_data=True,
        axes_scale="log",
        color="g",
        fill_contours=True,
    )
    _run_corner(
        N=5000,
        factor=0.5,
        seed=15,
        exp_data=True,
        axes_scale="log",
        color="b",
        fig=fig,
        fill_contours=True,
    )


@image_comparison(
    baseline_images=["bins"], remove_text=True, extensions=["png"]
)
def test_bins():
    _run_corner(bins=25)


@image_comparison(
    baseline_images=["bins_log"], remove_text=True, extensions=["png"]
)
def test_bins_log():
    _run_corner(exp_data=True, axes_scale="log", bins=25)


@image_comparison(
    baseline_images=["smooth"], remove_text=True, extensions=["png"]
)
def test_smooth():
    pytest.importorskip("scipy")
    _run_corner(bins=50, smooth=1.0)


@image_comparison(
    baseline_images=["smooth_log"], remove_text=True, extensions=["png"]
)
def test_smooth_log():
    pytest.importorskip("scipy")
    _run_corner(exp_data=True, axes_scale="log", bins=50, smooth=1.0)


@image_comparison(
    baseline_images=["smooth1d"], remove_text=True, extensions=["png"]
)
def test_smooth1d():
    pytest.importorskip("scipy")
    _run_corner(bins=50, smooth=1.0, smooth1d=1.0)


@image_comparison(
    baseline_images=["smooth1d_log"], remove_text=True, extensions=["png"]
)
def test_smooth1d_log():
    pytest.importorskip("scipy")
    _run_corner(
        exp_data=True, axes_scale="log", bins=50, smooth=1.0, smooth1d=1.0
    )


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


@image_comparison(baseline_images=["pandas"], extensions=["png"])
def test_pandas():
    _run_corner(pandas=True)


@image_comparison(
    baseline_images=["truths"], remove_text=True, extensions=["png"]
)
def test_truths():
    _run_corner(truths=[0.0, None, 0.15])


@image_comparison(
    baseline_images=["reverse_truths"], remove_text=True, extensions=["png"]
)
def test_reverse_truths():
    _run_corner(truths=[0.0, None, 0.15], reverse=True)


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
    baseline_images=["reverse_log"], remove_text=True, extensions=["png"]
)
def test_reverse_log():
    _run_corner(
        ndim=2,
        exp_data=True,
        axes_scale="log",
        range=[(1e4, 1e-4), (1e-5, 1e5)],
    )


@image_comparison(
    baseline_images=["extended_overplotting"],
    remove_text=True,
    extensions=["png"],
)
def test_extended_overplotting():
    # Test overplotting a more complex plot
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"]

    figure = _run_corner(ndim=4, reverse=False, labels=labels)

    # Set same results:
    ndim, nsamples = 4, 10000
    np.random.seed(1234)

    data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape(
        [4 * nsamples // 5, ndim]
    )
    mean = 4 * np.random.rand(ndim)

    value1 = mean
    # This is the empirical mean of the sample:
    value2 = np.mean(data1, axis=0)

    corner.overplot_lines(figure, value1, color="C1", reverse=False)
    corner.overplot_points(
        figure, value1[None], marker="s", color="C1", reverse=False
    )
    corner.overplot_lines(figure, value2, color="C2", reverse=False)
    corner.overplot_points(
        figure, value2[None], marker="s", color="C2", reverse=False
    )


@image_comparison(
    baseline_images=["reverse_overplotting"],
    remove_text=True,
    extensions=["png"],
)
def test_reverse_overplotting():
    # Test overplotting with a reversed plot
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"]

    figure = _run_corner(ndim=4, reverse=True, labels=labels)

    # Set same results:
    ndim, nsamples = 4, 10000
    np.random.seed(1234)

    data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape(
        [4 * nsamples // 5, ndim]
    )
    mean = 4 * np.random.rand(ndim)

    value1 = mean
    value2 = np.mean(data1, axis=0)

    corner.overplot_lines(figure, value1, color="C1", reverse=True)
    corner.overplot_points(
        figure, value1[None], marker="s", color="C1", reverse=True
    )
    corner.overplot_lines(figure, value2, color="C2", reverse=True)
    corner.overplot_points(
        figure, value2[None], marker="s", color="C2", reverse=True
    )


@image_comparison(
    baseline_images=["hist_bin_factor"], remove_text=True, extensions=["png"]
)
def test_hist_bin_factor():
    _run_corner(hist_bin_factor=4)


@image_comparison(
    baseline_images=["hist_bin_factor_log"],
    remove_text=True,
    extensions=["png"],
)
def test_hist_bin_factor_log():
    _run_corner(exp_data=True, axes_scale="log", hist_bin_factor=4)


@image_comparison(baseline_images=["arviz"], extensions=["png"])
def test_arviz():
    _run_corner(arviz=True)


@image_comparison(
    baseline_images=["range_fig_arg"], remove_text=True, extensions=["png"]
)
def test_range_fig_arg():
    fig = pl.figure()
    ranges = [(-1.1, 1), 0.8, (-1, 1)]
    _run_corner(N=100_000, range=ranges, fig=fig)


@image_comparison(baseline_images=["1d_fig_argument"], extensions=["png"])
def test_1d_fig_argument():
    fig = _run_corner(ndim=1, seed=0)
    _run_corner(ndim=1, seed=1, fig=fig)
