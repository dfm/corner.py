# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner", "hist2d"]
__version__ = "0.1.1"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
    "Adrian Price-Whelan @adrn",
    "Brendon Brewer @eggplantbren",
    "Ekta Patel @ekta1224",
    "Emily Rice @emilurice",
    "Geoff Ryan @geoffryan",
    "Kyle Barbary @kbarbary",
    "Phil Marshall @drphilmarshall",
    "Pierre Gratier @pirg",
]

import logging
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter


def corner(xs, weights=None, labels=None, show_titles=False, title_fmt=".2f",
           title_args=None, extents=None, truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=None, verbose=True,
           plot_contours=True, plot_datapoints=True, fig=None, **kwargs):
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

    weights : array_like (nsamples,)
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    labels : iterable (ndim,) (optional)
        A list of names for the dimensions.

    show_titles : bool (optional)
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string (optional)
        The format string for the quantiles given in titles.
        (default: `.2f`)

    title_args : dict (optional)
        Any extra keyword arguments to send to the `add_title` command.

    extents : iterable (ndim,) (optional)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds (extents) or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

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

    verbose : bool (optional)
        If true, print the values of the computed quantiles.

    plot_contours : bool (optional)
        Draw contours for dense regions of the plot.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    fig : matplotlib.Figure (optional)
        Overplot onto the provided figure object.

    """
    if quantiles is None:
        quantiles = []

    if title_args is None:
        title_args = {}

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError('weights must be 1-D')
        if xs.shape[1] != weights.shape[0]:
            raise ValueError('lengths of weights must match number of samples')

    # backwards-compatibility
    plot_contours = kwargs.get("smooth", plot_contours)

    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

        # Check for parameters that never change.
        m = np.array([e[0] == e[1] for e in extents], dtype=bool)
        if np.any(m):
            raise ValueError(("It looks like the parameter(s) in column(s) "
                              "{0} have no dynamic range. Please provide an "
                              "`extent` argument.")
                             .format(", ".join(map("{0}".format,
                                                   np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        for i in range(len(extents)):
            try:
                emin, emax = extents[i]
            except TypeError:
                q = [0.5 - 0.5*extents[i], 0.5 + 0.5*extents[i]]
                extents[i] = quantile(xs[i], q, weights=weights)

    for i, x in enumerate(xs):
        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.
        n, b, p = ax.hist(x, weights=weights, bins=kwargs.get("bins", 50),
                          range=extents[i], histtype="step",
                          color=kwargs.get("color", "k"))
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=kwargs.get("color", "k"))

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

            if show_titles:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84],
                                            weights=weights)
                q_m, q_p = q_50-q_16, q_84-q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if labels is not None:
                    title = "{0} = {1}".format(labels[i], title)

                # Add the title to the axis.
                ax.set_title(title, **title_args)

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

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]],
                   plot_contours=plot_contours,
                   plot_datapoints=plot_datapoints,
                   weights=weights, **kwargs)

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


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def hist2d(x, y, bins=20, range=None, weights=None, levels=None, ax=None,
           color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    if ax is None:
        ax = pl.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [rgba_color] + [list(rgba_color) for l in levels]
    for i, l in enumerate(levels):
        contour_cmap[i+1][-1] *= float(len(levels) - i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=range, weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if plot_contours or plot_density:
        ax.contourf(X1, Y1, H.T, [V[-1], H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X1, Y1, H.T, np.concatenate([[H.max()], V, [0]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X1, Y1, H.T, V, **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
