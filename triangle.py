# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner", "hist2d"]
__version__ = "0.2.0"

import logging
from corner import corner, hist2d

logging.warn("Deprecation Warning: 'triangle' has been renamed to 'corner'. "
             "This shim should continue to work but you should use 'import "
             "corner' in new code. "
             "https://github.com/dfm/corner.py")
