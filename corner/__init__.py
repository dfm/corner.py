# -*- coding: utf-8 -*-

__version__ = "2.0.2.dev0"
__author__ = "Dan Foreman-Mackey (foreman.mackey@gmail.com)"
__copyright__ = "Copyright 2013-2016 Daniel Foreman-Mackey and contributors"
__contributors__ = [
    # Alphabetical by first name.
    "Adrian Price-Whelan @adrn",
    "Brendon Brewer @eggplantbren",
    "Brigitta Sipocz @bsipocz",
    "Ekta Patel @ekta1224",
    "Emily Rice @emilurice",
    "Geoff Ryan @geoffryan",
    "Guillaume @ceyzeriat",
    "Gregory Ashton @ga7g08",
    "Hanno Rein @hannorein",
    "Jeremy Heyl @jsheyl",
    "Kelle Cruz @kelle",
    "Kyle Barbary @kbarbary",
    "Marco Tazzari @mtazzari",
    "Matt Pitkin @mattpitkin",
    "Phil Marshall @drphilmarshall",
    "Pierre Gratier @pirg",
    "Stephan Hoyer @shoyer",
    "Víctor Zabalza @zblz",
    "Will Vousden @willvousden",
    "Wolfgang Kerzendorf @wkerzendorf",
]
__bibtex__ = """@article{corner,
    Author = {Daniel Foreman-Mackey},
    Doi = {10.21105/joss.00024},
    Title = {corner.py: Scatterplot matrices in Python},
    Journal = {The Journal of Open Source Software},
    Year = 2016,
    Volume = 24,
    Url = {http://dx.doi.org/10.5281/zenodo.45906}
}"""

try:
    __CORNER_SETUP__
except NameError:
    __CORNER_SETUP__ = False

if not __CORNER_SETUP__:
    __all__ = ["corner", "hist2d", "quantile"]

    from .corner import corner, hist2d, quantile
