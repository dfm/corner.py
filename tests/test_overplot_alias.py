# -*- coding: utf-8 -*-

import numpy as np
import pytest
from matplotlib import pyplot as pl

import corner


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("linestyle", ["--", ":", "none"])
def test_overplot_points_linestyle_alias(reverse, linestyle):
    samples = np.random.default_rng(42).normal(size=(100, 2))
    points = np.array([[0.1, 0.4], [0.3, 0.2], [0.5, -0.1]])
    fig = corner.corner(samples, reverse=reverse)
    try:
        ax = fig.axes[1 if reverse else 2]
        corner.overplot_points(
            fig, points, reverse=reverse, ls=linestyle, c="C2", marker="s"
        )
        alias_line = ax.lines[-1]
        corner.overplot_points(
            fig,
            points,
            reverse=reverse,
            linestyle=linestyle,
            color="C2",
            marker="s",
        )
        full_line = ax.lines[-1]
        np.testing.assert_array_equal(alias_line.get_xdata(), points[:, 0])
        np.testing.assert_array_equal(alias_line.get_ydata(), points[:, 1])
        assert alias_line.get_linestyle() == full_line.get_linestyle()
        assert alias_line.get_color() == full_line.get_color()
        assert alias_line.get_marker() == full_line.get_marker()
    finally:
        pl.close(fig)


def test_overplot_points_default_linestyle():
    fig = corner.corner(np.random.default_rng(42).normal(size=(100, 2)))
    try:
        corner.overplot_points(fig, [[0.1, 0.2], [0.3, 0.4]])
        assert fig.axes[2].lines[-1].get_linestyle() == "None"
        assert fig.axes[2].lines[-1].get_marker() == "."
        with pytest.raises(TypeError, match="aliases"):
            corner.overplot_points(fig, [[0.1, 0.2]], ls="--", linestyle=":")
    finally:
        pl.close(fig)
