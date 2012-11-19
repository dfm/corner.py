__all__ = ["hist2d", "error_ellipse", "corner"]


__version__ = "0.0.1"


import numpy as np
import scipy.special as sp

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


def error_ellipse(mu, cov, ax=None, **kwargs):
    """
    Plot the error ellipse at a point given it's covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width=2 * np.sqrt(S[0]),
            height=2 * np.sqrt(S[1]),
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = pl.gca()
    ax.add_patch(ellipsePlot)


def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", pl.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 50)
    color = kwargs.pop("color", 'k')

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y))

    V = sp.erf(np.arange(0.5, 2.1, 0.5) / np.sqrt(2))
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

    X, Y = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
            rasterized=True)
    ax.contourf(X, Y, H.T, [V[-1], 0.],
        cmap=LinearSegmentedColormap.from_list("cmap", ([1] * 3, [1] * 3),
            N=2))
    ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
    ax.contour(X, Y, H.T, V, colors=color)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])


def corner(xs, labels=None, extents=None, truths=None, truth_color="#4682b4",
           **kwargs):
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
        ax.hist(x, bins=kwargs.get("bins", 50), range=extents[i],
                histtype="step", color=kwargs.get("color", "k"))
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Set up the axes.
        ax.set_xlim(extents[i])
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
            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]], **kwargs)

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
