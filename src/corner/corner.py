# -*- coding: utf-8 -*-

import copy
import logging
from collections.abc import Mapping

import numpy as np
from arviz.data import convert_to_dataset
from arviz.utils import _var_names, get_coords
from matplotlib import pyplot as pl
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# Support multiple versions of arviz
try:
    from arviz.plots.plot_utils import (
        make_label,
        xarray_to_ndarray,
        xarray_var_iter,
    )

    def _get_labels(plotters, labeller=None):
        return [
            make_label(var_name, selection)
            for var_name, selection, _ in plotters
        ]


except ImportError:
    from arviz.labels import BaseLabeller
    from arviz.sel_utils import xarray_to_ndarray, xarray_var_iter

    def _get_labels(plotters, labeller=None):
        if labeller is None:
            labeller = BaseLabeller()
        return [
            labeller.make_label_vert(var_name, sel, isel)
            for var_name, sel, isel, _ in plotters
        ]


__all__ = ["corner", "hist2d", "quantile", "overplot_lines", "overplot_points"]


def corner(
    data,
    bins=20,
    *,
    # Original corner parameters
    range=None,
    weights=None,
    color="k",
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
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    # Arviz parameters
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
    **hist2d_kwargs
):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an ``arviz.InferenceData`` object.
        Refer to documentation of ``arviz.convert_to_dataset`` for details.

    bins : int or array_like[ndim,]
        The number of bins to use in histograms, either as a fixed value for
        all dimensions, or as a list of integers for each dimension.

    group : str
        Specifies which InferenceData group should be plotted.  Defaults to
        ``'posterior'``.

    var_names : list
        Variables to be plotted, if ``None`` all variable are plotted. Prefix
        the variables by `~` when you want to exclude them from the plot.

    filter_vars : {``None``, ``"like"``, ``"regex"``}
        If ``None`` (default), interpret ``var_names`` as the real variables
        names. If ``"like"``, interpret ``var_names`` as substrings of the real
        variables names. If ``"regex"``, interpret ``var_names`` as regular
        expressions on the real variables names. A la ``pandas.filter``.

    coords : mapping
        Coordinates of ``var_names`` to be plotted. Passed to
        ``arviz.Dataset.sel``.

    divergences : bool
        If ``True`` divergences will be plotted in a different color, only if
        ``group`` is either ``'prior'`` or ``'posterior'``.

    divergences_kwargs : dict
        Any extra keyword arguments to send to the ``overplot_points`` when
        plotting the divergences.

    labeller : arviz.Labeller
        Class providing the method ``make_label_vert`` to generate the labels
        in the plot. Read the ArviZ label guide for more details and usage
        examples.

    weights : array_like[nsamples,]
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    color : str
        A ``matplotlib`` style color for all histograms.

    hist_bin_factor : float or array_like[ndim,]
        This is a factor (or list of factors, one for each dimension) that
        will multiply the bin specifications when making the 1-D histograms.
        This is generally used to increase the number of bins in the 1-D plots
        to provide more resolution.

    smooth, smooth1d : float
       The standard deviation for Gaussian kernel passed to
       `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
       respectively. If `None` (default), no smoothing is applied.

    labels : iterable (ndim,)
        A list of names for the dimensions. If a ``xs`` is a
        ``pandas.DataFrame``, labels will default to column names.

    label_kwargs : dict
        Any extra keyword arguments to send to the `set_xlabel` and
        `set_ylabel` methods. Note that passing the `labelpad` keyword
        in this dictionary will not have the desired effect. Use the
        `labelpad` keyword in this function instead.

    titles : iterable (ndim,)
        A list of titles for the dimensions. If `None` (default),
        uses labels as titles.

    show_titles : bool
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string
        The format string for the quantiles given in titles. If you explicitly
        set ``show_titles=True`` and ``title_fmt=None``, the labels will be
        shown as the titles. (default: ``.2f``)

    title_kwargs : dict
        Any extra keyword arguments to send to the `set_title` command.

    range : iterable (ndim,)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,)
        A list of reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    truth_color : str
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool
        If true, print the values of the computed quantiles.

    plot_contours : bool
        Draw contours for dense regions of the plot.

    use_math_text : bool
        If true, then axis tick labels for very large or small exponents will
        be displayed as powers of 10 rather than using `e`.

    reverse : bool
        If true, plot the corner plot starting in the upper-right corner
        instead of the usual bottom-left corner

    labelpad : float
        Padding between the axis and the x- and y-labels in units of the
        fraction of the axis from the lower left

    max_n_ticks: int
        Maximum number of ticks to try to use

    top_ticks : bool
        If true, label the top ticks of each axis

    fig : matplotlib.Figure
        Overplot onto the provided figure object, which must either have no
        axes yet, or ``ndim * ndim`` axes already present.  If not set, the
        plot will be drawn on a newly created figure.

    hist_kwargs : dict
        Any extra keyword arguments to send to the 1-D histogram plots.

    **hist2d_kwargs
        Any remaining keyword arguments are sent to :func:`corner.hist2d` to
        generate the 2-D histogram plots.

    """
    is_np = False
    if isinstance(data, np.ndarray):
        is_np = True
        if data.ndim == 1:
            data = data[None, :, :]
        elif data.ndim == 2:
            data = data[None, :, :]
        elif data.ndim != 3:
            raise ValueError("invalid input dimensions")
    if data.__class__.__name__ == "DataFrame":
        logging.warning(
            "Pandas support in corner is deprecated; use ArviZ directly"
        )
        data = {k: np.asarray(data[k])[None] for k in list(data.columns)}

    if coords is None:
        coords = {}

    # Get posterior draws and combine chains
    dataset = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, dataset, filter_vars)
    plotters = list(
        xarray_var_iter(
            get_coords(dataset, coords), var_names=var_names, combined=True
        )
    )
    if labels is None and not is_np:
        labels = _get_labels(plotters, labeller=labeller)
    if var_names is None:
        var_names = dataset.data_vars

    divergent_data = None
    diverging_mask = None

    # Assigning divergence group based on group param
    if group == "posterior":
        divergent_group = "sample_stats"
    elif group == "prior":
        divergent_group = "sample_stats_prior"
    else:
        divergences = False

    # Reformat truths and titles as lists if they are mappings
    if isinstance(truths, Mapping):
        truths = np.concatenate(
            [np.asarray(truths[k]).flatten() for k in var_names]
        )
    if isinstance(titles, Mapping):
        titles = np.concatenate(
            [np.asarray(titles[k]).flatten() for k in var_names]
        )

    # Coerce the samples into the expected format
    samples = np.stack([x[-1].flatten() for x in plotters], axis=-1)
    fig = _corner_backend(
        samples,
        bins=bins,
        range=range,
        weights=weights,
        color=color,
        hist_bin_factor=hist_bin_factor,
        smooth=smooth,
        smooth1d=smooth1d,
        labels=labels,
        label_kwargs=label_kwargs,
        titles=titles,
        show_titles=show_titles,
        title_fmt=title_fmt,
        title_kwargs=title_kwargs,
        truths=truths,
        truth_color=truth_color,
        scale_hist=scale_hist,
        quantiles=quantiles,
        verbose=verbose,
        fig=fig,
        max_n_ticks=max_n_ticks,
        top_ticks=top_ticks,
        use_math_text=use_math_text,
        reverse=reverse,
        labelpad=labelpad,
        hist_kwargs=hist_kwargs,
        **hist2d_kwargs,
    )

    # Get diverging draws and combine chains
    if divergences:
        if hasattr(data, divergent_group) and hasattr(
            getattr(data, divergent_group), "diverging"
        ):
            divergent_data = convert_to_dataset(data, group=divergent_group)
            _, diverging_mask = xarray_to_ndarray(
                divergent_data, var_names=("diverging",), combined=True
            )
            diverging_mask = np.squeeze(diverging_mask)
            if divergences_kwargs is None:
                divergences_kwargs = {"color": "C1", "ms": 1}
            overplot_points(fig, samples[diverging_mask], **divergences_kwargs)

    return fig


def _corner_backend(
    xs,
    bins=20,
    range=None,
    weights=None,
    color="k",
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
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    **hist2d_kwargs
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

    # Create a new figure if one wasn't provided.
    new_fig = True
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        new_fig = False
        axes = _get_fig_axes(fig, K)

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    # Parse the parameter ranges.
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
            ax = axes
        else:
            if reverse:
                ax = axes[K - i - 1, K - i - 1]
            else:
                ax = axes[i, i]

        # Plot the histograms.
        if smooth1d is None:
            bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
            n, _, _ = ax.hist(
                x,
                bins=bins_1d,
                weights=weights,
                range=np.sort(range[i]),
                **hist_kwargs,
            )
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(
                x, bins=bins[i], weights=weights, range=np.sort(range[i])
            )
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
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
                q_16, q_50, q_84 = quantile(
                    x, [0.16, 0.5, 0.84], weights=weights
                )
                q_m, q_p = q_50 - q_16, q_84 - q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

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
        _set_xlim(new_fig, ax, range[i])
        if scale_hist:
            maxn = np.max(n)
            _set_ylim(new_fig, ax, [-0.1 * maxn, 1.1 * maxn])

        else:
            _set_ylim(new_fig, ax, [0, 1.1 * np.max(n)])

        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            [lbl.set_rotation(45) for lbl in ax.get_xticklabels()]
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
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text)
            )

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
                weights=weights,
                color=color,
                smooth=smooth,
                bins=[bins[j], bins[i]],
                new_fig=new_fig,
                **hist2d_kwargs,
            )

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
                ax.yaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4 + labelpad)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3 + labelpad, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3 - labelpad, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )

    if truths is not None:
        overplot_lines(fig, truths, color=truth_color)
        overplot_points(
            fig,
            [[np.nan if t is None else t for t in truths]],
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
    **kwargs
):

    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    quiet : bool
        If true, suppress warnings for small datasets.

    levels : array_like
        The contour levels to draw.

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
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)]
    )

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            range=list(map(np.sort, range)),
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
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
            cmap=white_cmap,
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

    _set_xlim(new_fig, ax, range[0])
    _set_ylim(new_fig, ax, range[1])


def overplot_lines(fig, xs, **kwargs):
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

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.axvline``
        method.

    """
    K = len(xs)
    axes = _get_fig_axes(fig, K)
    for k1 in range(K):
        if xs[k1] is not None:
            axes[k1, k1].axvline(xs[k1], **kwargs)
        for k2 in range(k1 + 1, K):
            if xs[k1] is not None:
                axes[k2, k1].axvline(xs[k1], **kwargs)
            if xs[k2] is not None:
                axes[k2, k1].axhline(xs[k2], **kwargs)


def overplot_points(fig, xs, **kwargs):
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

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.plot``
        method.

    """
    kwargs["marker"] = kwargs.pop("marker", ".")
    kwargs["linestyle"] = kwargs.pop("linestyle", "none")
    xs = _parse_input(xs)
    K = len(xs)
    axes = _get_fig_axes(fig, K)
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
        return fig.subplots(K, K)
    try:
        return np.array(fig.axes).reshape((K, K))
    except ValueError:
        raise ValueError(
            (
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}"
            ).format(len(fig.axes), K)
        )


def _set_xlim(new_fig, ax, new_xlim):
    if new_fig:
        return ax.set_xlim(new_xlim)
    xlim = ax.get_xlim()
    return ax.set_xlim([min(xlim[0], new_xlim[0]), max(xlim[1], new_xlim[1])])


def _set_ylim(new_fig, ax, new_ylim):
    if new_fig:
        return ax.set_ylim(new_ylim)
    ylim = ax.get_ylim()
    return ax.set_ylim([min(ylim[0], new_ylim[0]), max(ylim[1], new_ylim[1])])
