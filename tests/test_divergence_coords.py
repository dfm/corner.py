# -*- coding: utf-8 -*-

import numpy as np
import pytest
from matplotlib import pyplot as pl

import corner


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "coords",
    [
        {"chain": [1]},
        {"draw": [1, 3, 6, 8]},
        {"chain": [1, 0]},
        {"draw": [8, 6, 3, 1]},
        {"chain": [1], "draw": [1, 3, 6, 8]},
        {"school": ["b"]},
        {},
    ],
)
def test_divergences_follow_sample_coords(group, reverse, coords):
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 10))
    y = rng.normal(size=(2, 10, 2))
    mask = np.zeros((2, 10), dtype=bool)
    mask[0, [1, 3]] = True
    mask[1, [6, 8]] = True
    stats_group = (
        "sample_stats" if group == "posterior" else "sample_stats_prior"
    )
    groups = {group: {"x": x, "y": y}, stats_group: {"diverging": mask}}
    conversion_kwargs = {
        "coords": {"school": ["a", "b"]},
        "dims": {"y": ["school"]},
    }
    try:
        data = az.from_dict(**groups, **conversion_kwargs)
    except TypeError:
        data = az.from_dict(groups, **conversion_kwargs)
    fig = corner.corner(
        data,
        group=group,
        coords=coords,
        divergences=True,
        reverse=reverse,
        plot_contours=False,
    )
    try:
        # Build the expected divergent points directly from input arrays.
        chains = coords.get("chain", [0, 1])
        draws = coords.get("draw", list(range(10)))
        schools = [1] if "school" in coords else [0, 1]
        selected_mask = mask[np.ix_(chains, draws)].ravel()
        expected_x = x[np.ix_(chains, draws)].ravel()[selected_mask]
        ndim = 1 + len(schools)
        axes = np.array(fig.axes).reshape(ndim, ndim)
        for i, school in enumerate(schools, 1):
            expected_y = y[:, :, school][np.ix_(chains, draws)].ravel()[
                selected_mask
            ]
            ax = axes[ndim - i - 1, ndim - 1] if reverse else axes[i, 0]
            line = ax.lines[-1]
            np.testing.assert_array_equal(line.get_xdata(), expected_x)
            np.testing.assert_array_equal(line.get_ydata(), expected_y)
    finally:
        pl.close(fig)
