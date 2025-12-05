# -*- coding: utf-8 -*-

__all__ = [
    "corner", "hist2d", "quantile", "overplot_lines", "overplot_points",
    "axis_from_param_indices",
    "param_indices_from_axis"
]

from corner.core import (
    hist2d, overplot_lines, overplot_points, quantile, axis_from_param_indices, param_indices_from_axis
)

from corner.corner import corner
from corner.version import version as __version__
