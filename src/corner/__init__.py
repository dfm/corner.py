# -*- coding: utf-8 -*-

__all__ = [
    "corner",
    "hist2d",
    "quantile",
    "overplot_lines",
    "overplot_points",
    "axis_from_param_indices",
    "param_indices_from_axis",
]

from corner.core import (
    axis_from_param_indices,
    hist2d,
    overplot_lines,
    overplot_points,
    param_indices_from_axis,
    quantile,
)
from corner.corner import corner
from corner.version import version as __version__
