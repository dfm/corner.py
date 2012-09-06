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


def corner(xs, labels=None, **kwargs):
    factor = 2.5
    dim = factor * len(xs)
    fig = pl.figure(figsize=(dim, dim))
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95,
            wspace=0.04, hspace=0.04)

    for i, x in enumerate(xs):
        # Plot the histograms.
        ax = fig.add_subplot(len(xs), len(xs), i * (len(xs) + 1) + 1)
        ax.hist(x, bins=kwargs.get("bins", 50), histtype="step",
                color=kwargs.get("color", "k"))

        # Set up the axes.
        ax.set_xlim([x.min(), x.max()])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < len(xs) - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(35) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs[:i]):
            ax = fig.add_subplot(len(xs), len(xs),
                    (i * len(xs) + j) + 1)
            hist2d(y, x, ax=ax, **kwargs)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < len(xs) - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(35) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.4, 0.5)

    return fig
