# -*- coding: utf-8 -*-

__all__ = ["corner", "hist2d", "quantile", "overplot_lines", "overplot_points"]

from corner.core import hist2d, overplot_lines, overplot_points, get_axis_by_index, quantile
from corner.corner import corner
from corner.version import version as __version__
