# -*- coding: utf-8 -*-

__all__ = ["corner", "hist2d", "quantile", "overplot_lines", "overplot_points"]

from corner.core import (
    get_axis_by_index,
    hist2d,
    overplot_lines,
    overplot_points,
    quantile,
)
from corner.corner import corner
from corner.version import version as __version__
