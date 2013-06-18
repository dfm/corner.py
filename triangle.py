#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner", "hist2d", "error_ellipse"]
__version__ = "0.0.5"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
__contributors__ = [    # Alphabetical by first name.
                        "Adrian Price-Whelan @adrn",
                        "Brendon Brewer @eggplantbren",
                        "Ekta Patel @ekta1224",
                        "Emily Rice @emilurice",
                        "Geoff Ryan @geoffryan",
                        "Phil Marshall @drphilmarshall",
                        "Pierre Gratier @pirg",
                   ]

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


def corner(xs, labels=None, extents=None, truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=[], plot_contours=True, 
           plot_datapoints=True, **kwargs):
    """
    Make a *sick* corner plot showing the projections of a data set in a 
    multi-dimensional space. kwargs are passed to hist2d() or used for 
    `matplotlib` styling.
    
    Parameters
    ----------
    xs : array_like (nsamples, ndim)
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    labels : iterable (ndim,) (optional)
        A list of names for the dimensions.
    
    extents : iterable (ndim,) (optional)
        A list of length 2 tuples containing lower and upper bounds (extents)
        for each dimension, e.g., [(0.,10.), (1.,5), etc.]
    
    truths : iterable (ndim,) (optional)
        A list of reference values to indicate on the plots.

    truth_color : str (optional)
        A ``matplotlib`` style color for the ``truths`` makers.
    
    scale_hist : bool (optional)
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?
    
    quantiles : iterable (optional)
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.
    
    plot_contours : bool (optional)
        Draw contours for dense regions of the plot.
    
    plot_datapoints : bool (optional)
        Draw the individual data points.

    """

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"
    
    # backwards-compatibility
    plot_contours = kwargs.get("smooth", plot_contours)
    
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.05 * factor  # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = pl.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

    for i, x in enumerate(xs):
        # Plot the histograms.
        ax = fig.add_subplot(K, K, i * (K + 1) + 1)
        n, b, p = ax.hist(x, bins=kwargs.get("bins", 50), range=extents[i],
                histtype="step", color=kwargs.get("color", "k"))
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            xsorted = sorted(x)
            for q in quantiles:
                ax.axvline(xsorted[int(q * len(xsorted))], ls="dashed",
                           color=kwargs.get("color", "k"))
            print quantiles # Print quantiles if wanted.
                
        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs[:i]):
            ax = fig.add_subplot(K, K, (i * K + j) + 1)
            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]], 
                   plot_contours=plot_contours,
                   plot_datapoints=plot_datapoints,
                   **kwargs)

            if truths is not None:
                ax.plot(truths[j], truths[i], "s", color=truth_color)
                ax.axvline(truths[j], color=truth_color)
                ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width=2 * np.sqrt(S[0]) * factor,
            height=2 * np.sqrt(S[1]) * factor,
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = pl.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", pl.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 50)
    color = kwargs.pop("color", "k")
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y))

    V = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]
    
    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        if plot_contours:
            ax.contourf(X1, Y1, H.T, [V[-1], H.max()],
                cmap=LinearSegmentedColormap.from_list("cmap", ([1] * 3, [1] * 3),
                   N=2), antialiased=False)

    if plot_contours:
        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
        ax.contour(X1, Y1, H.T, V, colors=color)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
