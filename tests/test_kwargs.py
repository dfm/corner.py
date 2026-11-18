import numpy as np
import pytest
from matplotlib import colors
from matplotlib import pyplot as plt

import corner


@pytest.mark.parametrize(
    "smoothing", [(None, None), (1, 1), (None, 1), (1, None)]
)
@pytest.mark.parametrize("explicit_color", [None, "green"])
def test_reuse_hist_kwargs(smoothing, explicit_color):
    samples = np.random.default_rng(1234).normal(size=(500, 2))
    hist_kwargs = {"linewidth": 2}
    if explicit_color is not None:
        hist_kwargs["color"] = explicit_color
    original = hist_kwargs.copy()
    fig = None
    try:
        for smooth, color in zip(smoothing, ["red", "blue"]):
            fig = corner.corner(
                samples,
                fig=fig,
                color=color,
                hist_kwargs=hist_kwargs,
                smooth1d=smooth,
                plot_contours=False,
            )
            ax = fig.axes[0]
            actual = (
                ax.patches[-1].get_edgecolor()
                if smooth is None
                else ax.lines[-1].get_color()
            )
            assert colors.to_rgba(actual) == colors.to_rgba(
                explicit_color or color
            )
        assert hist_kwargs == original
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "keyword", ["data_kwargs", "contour_kwargs", "contourf_kwargs"]
)
def test_reuse_hist2d_kwargs(keyword):
    samples = np.random.default_rng(1234).normal(size=(1000, 2))
    options = {"alpha": 0.4}
    original = options.copy()
    fig, ax = plt.subplots()
    try:
        for color in ["red", "blue"]:
            corner.hist2d(
                *samples.T,
                ax=ax,
                color=color,
                fill_contours=True,
                **{keyword: options},
            )
            if keyword == "data_kwargs":
                actual = ax.lines[-1].get_color()
            elif keyword == "contour_kwargs":
                actual = ax.collections[-1].get_edgecolor()[0]
            else:
                filled = [c for c in ax.collections if len(c.get_facecolor())]
                actual = filled[-1].get_facecolor()[-1]
            np.testing.assert_allclose(
                colors.to_rgba(actual)[:3], colors.to_rgba(color)[:3]
            )
        assert options == original
    finally:
        plt.close(fig)
