# -*- coding: utf-8 -*-

import numpy as np
import pytest
from matplotlib import pyplot as pl

import corner


@pytest.mark.parametrize(
    "var_names, filter_vars, expected",
    [
        (["~theta"], None, ["mu", "tau"]),
        (["~the"], "like", ["mu", "tau"]),
        (["~^the.*"], "regex", ["mu", "tau"]),
        ("mu", None, ["mu"]),
        ("tau", "like", ["tau"]),
        ("^mu$", "regex", ["mu"]),
        (["theta", "mu"], None, ["theta", "mu"]),
        (["tau"], "like", ["tau"]),
        (["^mu$"], "regex", ["mu"]),
    ],
)
def test_arviz_variable_selection(var_names, filter_vars, expected):
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(42)
    posterior = {
        name: rng.normal(size=(2, 100)) for name in ["mu", "tau", "theta"]
    }
    try:
        data = az.from_dict(posterior=posterior)
    except TypeError:
        data = az.from_dict({"posterior": posterior})
    fig = corner.corner(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        quantiles=[0.5],
        plot_contours=False,
    )
    try:
        ndim = len(expected)
        assert len(fig.axes) == ndim**2
        for i, name in enumerate(expected):
            # The median line checks selection against the original draws.
            ax = fig.axes[i * ndim + i]
            np.testing.assert_allclose(
                ax.lines[0].get_xdata(), np.median(posterior[name])
            )
    finally:
        pl.close(fig)
