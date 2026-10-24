import matplotlib.pyplot as plt
import numpy as np
import pytest

import corner


@pytest.mark.parametrize("convert", [list, tuple])
@pytest.mark.parametrize("explicit_range", [False, True])
def test_hist2d_array_like_samples(convert, explicit_range):
    x = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0.7, 0.1, 0.6, 0.4, 0.2])
    kwargs = dict(bins=4, plot_contours=False, plot_datapoints=False)
    if explicit_range:
        kwargs["range"] = [(0, 1), (0, 1)]
    fig, (expected, actual) = plt.subplots(1, 2)
    try:
        corner.hist2d(x, y, ax=expected, **kwargs)
        corner.hist2d(convert(x), convert(y), ax=actual, **kwargs)
        np.testing.assert_array_equal(actual.get_xlim(), expected.get_xlim())
        np.testing.assert_array_equal(actual.get_ylim(), expected.get_ylim())
        assert len(actual.collections) == len(expected.collections)
        for left, right in zip(actual.collections, expected.collections):
            np.testing.assert_array_equal(left.get_array(), right.get_array())
            assert len(left.get_paths()) == len(right.get_paths())
            for p, q in zip(left.get_paths(), right.get_paths()):
                np.testing.assert_array_equal(p.vertices, q.vertices)
    finally:
        plt.close(fig)
