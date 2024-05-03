# -*- coding: utf-8 -*-

__all__ = [
    "corner_impl",
    "hist2d",
    "quantile",
    "overplot_lines",
    "overplot_points",
]

import copy
import logging

import matplotlib
import numpy as np
from matplotlib import pyplot as pl
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullLocator,
    ScalarFormatter,
)

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def corner_impl(
    xs,
    bins=20,
    range=None,
    axes_scale="linear",
    weights=None,
    color=None,
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    labels=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    title_quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    **hist2d_kwargs,
):
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # If no separate titles are set, copy the axis labels
    if titles is None:
        titles = labels

    # deal with title quantiles so they much quantiles unless desired otherwise
    if title_quantiles is None:
        if len(quantiles) > 0:
            title_quantiles = quantiles
        else:
            # a default for when quantiles not supplied.
            title_quantiles = [0.16, 0.5, 0.84]

    if show_titles and len(title_quantiles) != 3:
        raise ValueError(
            "'title_quantiles' must contain exactly three values; "
            "pass a length-3 list or array using the 'title_quantiles' argument"
        )

    # Deal with 1D sample lists.
    xs = _parse_input(xs)
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Make axes_scale into a list if necessary, otherwise check length
    if isinstance(axes_scale, str):
        axes_scale = [axes_scale] * K
    else:
        assert (
            len(axes_scale) == K
        ), "'axes_scale' should contain as many elements as data dimensions"

    # Create a new figure if one wasn't provided.
    new_fig = True
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        axes, new_fig = _get_fig_axes(fig, K)

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    # Parse the parameter ranges.
    force_range = False
    if range is None:
        if "extents" in hist2d_kwargs:
            logging.warning(
                "Deprecated keyword argument 'extents'. "
                "Use 'range' instead."
            )
            range = hist2d_kwargs.pop("extents")
        else:
            range = [[x.min(), x.max()] for x in xs]

            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in range], dtype=bool)
            if np.any(m):
                raise ValueError(
                    (
                        "It looks like the parameter(s) in "
                        "column(s) {0} have no dynamic range. "
                        "Please provide a `range` argument."
                    ).format(
                        ", ".join(map("{0}".format, np.arange(len(m))[m]))
                    )
                )

    else:
        force_range = True

        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5 * range[i], 0.5 + 0.5 * range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")
    try:
        hist_bin_factor = [float(hist_bin_factor) for _ in range]
    except TypeError:
        if len(hist_bin_factor) != len(range):
            raise ValueError(
                "Dimension mismatch between hist_bin_factor and " "range"
            )

    # Set up the default plotting arguments.
    if color is None:
        color = matplotlib.rcParams["ytick.color"]

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = (
                axes if not isinstance(axes, np.ndarray) else axes.flatten()[0]
            )
        else:
            if reverse:
                ax = axes[K - i - 1, K - i - 1]
            else:
                ax = axes[i, i]

        # Plot the histograms.
        n_bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
        if axes_scale[i] == "linear":
            bins_1d = np.linspace(min(range[i]), max(range[i]), n_bins_1d + 1)
        elif axes_scale[i] == "log":
            bins_1d = np.logspace(
                np.log10(min(range[i])), np.log10(max(range[i])), n_bins_1d + 1
            )
        else:
            raise ValueError(
                "Scale "
                + axes_scale[i]
                + "for dimension "
                + str(i)
                + "not supported. Use 'linear' or 'log'"
            )
        if smooth1d is None:
            n, _, _ = ax.hist(x, bins=bins_1d, weights=weights, **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, _ = np.histogram(x, bins=bins_1d, weights=weights)
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(bins_1d[:-1], bins_1d[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_lo, q_mid, q_hi = quantile(
                    x, title_quantiles, weights=weights
                )
                q_m, q_p = q_mid - q_lo, q_hi - q_mid

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if titles is not None:
                    title = "{0} = {1}".format(titles[i], title)

            elif titles is not None:
                title = "{0}".format(titles[i])

            if title is not None:
                if reverse:
                    if "pad" in title_kwargs.keys():
                        title_kwargs_new = copy.copy(title_kwargs)
                        del title_kwargs_new["pad"]
                        title_kwargs_new["labelpad"] = title_kwargs["pad"]
                    else:
                        title_kwargs_new = title_kwargs

                    ax.set_xlabel(title, **title_kwargs_new)
                else:
                    ax.set_title(title, **title_kwargs)

        # Set up the axes.
        _set_xlim(force_range, new_fig, ax, range[i])
        ax.set_xscale(axes_scale[i])
        if scale_hist:
            maxn = np.max(n)
            _set_ylim(force_range, new_fig, ax, [-0.1 * maxn, 1.1 * maxn])

        else:
            _set_ylim(force_range, new_fig, ax, [0, 1.1 * np.max(n)])

        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            if axes_scale[i] == "linear":
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
            elif axes_scale[i] == "log":
                ax.xaxis.set_major_locator(LogLocator(numticks=max_n_ticks))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
            else:
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
            if labels is not None:
                if reverse:
                    if "labelpad" in label_kwargs.keys():
                        label_kwargs_new = copy.copy(label_kwargs)
                        del label_kwargs_new["labelpad"]
                        label_kwargs_new["pad"] = label_kwargs["labelpad"]
                    else:
                        label_kwargs_new = label_kwargs
                    ax.set_title(
                        labels[i],
                        position=(0.5, 1.3 + labelpad),
                        **label_kwargs_new,
                    )

                else:
                    ax.set_xlabel(labels[i], **label_kwargs)
                    ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

            # use MathText for axes ticks
            if axes_scale[i] == "linear":
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )
            elif axes_scale[i] == "log":
                ax.xaxis.set_major_formatter(LogFormatterMathtext())

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K - i - 1, K - j - 1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(
                y,
                x,
                ax=ax,
                range=[range[j], range[i]],
                axes_scale=[axes_scale[j], axes_scale[i]],
                weights=weights,
                color=color,
                smooth=smooth,
                bins=[bins[j], bins[i]],
                new_fig=new_fig,
                force_range=force_range,
                **hist2d_kwargs,
            )

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                if axes_scale[j] == "linear":
                    ax.xaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                elif axes_scale[j] == "log":
                    ax.xaxis.set_major_locator(
                        LogLocator(numticks=max_n_ticks)
                    )

                if axes_scale[i] == "linear":
                    ax.yaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                elif axes_scale[i] == "log":
                    ax.yaxis.set_major_locator(
                        LogLocator(numticks=max_n_ticks)
                    )

            if i < K - 1:
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4 + labelpad)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

                # use MathText for axes ticks
                if axes_scale[j] == "linear":
                    ax.xaxis.set_major_formatter(
                        ScalarFormatter(useMathText=use_math_text)
                    )
                elif axes_scale[j] == "log":
                    ax.xaxis.set_major_formatter(LogFormatterMathtext())

            if j > 0:
                ax.set_yticklabels([])
                ax.set_yticklabels([], minor=True)
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                [l.set_rotation(45) for l in ax.get_yticklabels(minor=True)]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3 + labelpad, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3 - labelpad, 0.5)

                # use MathText for axes ticks
                if axes_scale[i] == "linear":
                    ax.yaxis.set_major_formatter(
                        ScalarFormatter(useMathText=use_math_text)
                    )
                elif axes_scale[i] == "log":
                    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    if truths is not None:
        overplot_lines(fig, truths, reverse=reverse, color=truth_color)
        overplot_points(
            fig,
            [[np.nan if t is None else t for t in truths]],
            reverse=reverse,
            marker="s",
            color=truth_color,
        )

    return fig


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


def hist2d(
    x,
    y,
    bins=20,
    range=None,
    axes_scale=["linear", "linear"],
    weights=None,
    levels=None,
    smooth=None,
    ax=None,
    color=None,
    quiet=False,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    no_fill_contours=False,
    fill_contours=False,
    contour_kwargs=None,
    contourf_kwargs=None,
    data_kwargs=None,
    pcolor_kwargs=None,
    new_fig=True,
    force_range=False,
    **kwargs,
):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    axes_scale : iterable (2,)
        Scale (``"linear"``, ``"log"``) to use for each dimension.

    quiet : bool
        If true, suppress warnings for small datasets.

    levels : array_like
        The contour levels to draw.
        If None, (0.5, 1, 1.5, 2)-sigma equivalent contours are drawn,
        i.e., containing 11.8%, 39.3%, 67.5% and 86.4% of the samples.
        See https://corner.readthedocs.io/en/latest/pages/sigmas/

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.

    """
    if ax is None:
        ax = pl.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warning(
                "Deprecated keyword argument 'extent'. Use 'range' instead."
            )
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = matplotlib.rcParams["ytick.color"]

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the base color of the axis (background color)
    base_color = ax.get_facecolor()

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, colorConverter.to_rgba(base_color, alpha=0.0)]
    )

    # This color map is used to hide the points at the high density areas.
    base_cmap = LinearSegmentedColormap.from_list(
        "base_cmap", [base_color, base_color], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")

    # We'll make the 2D histogram to directly estimate the density.
    bins_2d = []
    if axes_scale[0] == "linear":
        bins_2d.append(np.linspace(min(range[0]), max(range[0]), bins[0] + 1))
    elif axes_scale[0] == "log":
        bins_2d.append(
            np.logspace(
                np.log10(min(range[0])),
                np.log10(max(range[0])),
                bins[0] + 1,
            )
        )

    if axes_scale[1] == "linear":
        bins_2d.append(np.linspace(min(range[1]), max(range[1]), bins[1] + 1))
    elif axes_scale[1] == "log":
        bins_2d.append(
            np.logspace(
                np.log10(min(range[1])),
                np.log10(max(range[1])),
                bins[1] + 1,
            )
        )

    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins_2d,
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )
    if H.sum() == 0:
        raise ValueError(
            "It looks like the provided 'range' is not valid "
            "or the sample is empty."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    if plot_contours or plot_density:
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
            except IndexError:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        if np.any(m) and not quiet:
            logging.warning("Too few points to create valid contours")
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()

        # Compute the bin centers.
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        # Extend the array for the sake of the contours at the plot edges.
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate(
            [
                X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                X1,
                X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
            ]
        )
        Y2 = np.concatenate(
            [
                Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                Y1,
                Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
            ]
        )

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.min(), H.max()],
            cmap=base_cmap,
            antialiased=False,
        )

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get(
            "antialiased", False
        )
        ax.contourf(
            X2,
            Y2,
            H2.T,
            np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
            **contourf_kwargs,
        )

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        if pcolor_kwargs is None:
            pcolor_kwargs = dict()
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap, **pcolor_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    _set_xlim(force_range, new_fig, ax, range[0])
    _set_ylim(force_range, new_fig, ax, range[1])
    ax.set_xscale(axes_scale[0])
    ax.set_yscale(axes_scale[1])


def overplot_lines(fig, xs, reverse=False, **kwargs):
    """
    Overplot lines on a figure generated by ``corner.corner``

    Parameters
    ----------
    fig : Figure
        The figure generated by a call to :func:`corner.corner`.

    xs : array_like[ndim]
       The values where the lines should be plotted. This must have ``ndim``
       entries, where ``ndim`` is compatible with the :func:`corner.corner`
       call that originally generated the figure. The entries can optionally
       be ``None`` to omit the line in that axis.

    reverse: bool
       A boolean flag that should be set to 'True' if the corner plot itself
       was plotted with 'reverse=True'.

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.axvline``
        method.

    """
    K = len(xs)
    axes, _ = _get_fig_axes(fig, K)
    if reverse:
        for k1 in range(K):
            if xs[k1] is not None:
                axes[K - k1 - 1, K - k1 - 1].axvline(xs[k1], **kwargs)
            for k2 in range(k1 + 1, K):
                if xs[k1] is not None:
                    axes[K - k2 - 1, K - k1 - 1].axvline(xs[k1], **kwargs)
                if xs[k2] is not None:
                    axes[K - k2 - 1, K - k1 - 1].axhline(xs[k2], **kwargs)

    else:
        for k1 in range(K):
            if xs[k1] is not None:
                axes[k1, k1].axvline(xs[k1], **kwargs)
            for k2 in range(k1 + 1, K):
                if xs[k1] is not None:
                    axes[k2, k1].axvline(xs[k1], **kwargs)
                if xs[k2] is not None:
                    axes[k2, k1].axhline(xs[k2], **kwargs)


def overplot_points(fig, xs, reverse=False, **kwargs):
    """
    Overplot points on a figure generated by ``corner.corner``

    Parameters
    ----------
    fig : Figure
        The figure generated by a call to :func:`corner.corner`.

    xs : array_like[nsamples, ndim]
       The coordinates of the points to be plotted. This must have an ``ndim``
       that is compatible with the :func:`corner.corner` call that originally
       generated the figure.

    reverse: bool
       A boolean flag that should be set to 'True' if the corner plot itself
       was plotted with 'reverse=True'.

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.plot``
        method.

    """
    kwargs["marker"] = kwargs.pop("marker", ".")
    kwargs["linestyle"] = kwargs.pop("linestyle", "none")
    xs = _parse_input(xs)
    K = len(xs)
    axes, _ = _get_fig_axes(fig, K)
    if reverse:
        for k1 in range(K):
            for k2 in range(k1):
                axes[K - k1 - 1, K - k2 - 1].plot(xs[k2], xs[k1], **kwargs)

    else:
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                axes[k2, k1].plot(xs[k1], xs[k2], **kwargs)


def _parse_input(xs):
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    return xs


def _get_fig_axes(fig, K):
    if not fig.axes:
        return fig.subplots(K, K), True
    try:
        axarr = np.array(fig.axes).reshape((K, K))
        return axarr.item() if axarr.size == 1 else axarr.squeeze(), False
    except ValueError:
        raise ValueError(
            (
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}"
            ).format(len(fig.axes), K)
        )


def _set_xlim(force, new_fig, ax, new_xlim):
    if force or new_fig:
        return ax.set_xlim(new_xlim)
    xlim = ax.get_xlim()
    return ax.set_xlim([min(xlim[0], new_xlim[0]), max(xlim[1], new_xlim[1])])


def _set_ylim(force, new_fig, ax, new_ylim):
    if force or new_fig:
        return ax.set_ylim(new_ylim)
    ylim = ax.get_ylim()
    return ax.set_ylim([min(ylim[0], new_ylim[0]), max(ylim[1], new_ylim[1])])
