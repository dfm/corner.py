# -*- coding: utf-8 -*-

__all__ = "corner"

import logging

import numpy as np

from corner.core import corner_impl

try:
    from corner.arviz_corner import arviz_corner
except ImportError:
    arviz_corner = None


def corner(
    data,
    bins=20,
    *,
    # Original corner parameters
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
    title_quantiles=None,
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
    **hist2d_kwargs,
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
        A list of names for the dimensions.

        .. deprecated:: 2.2.1
            If a ``xs`` is a ``pandas.DataFrame`` *and* ArviZ is installed,
            labels will default to column names.
            This behavior will be removed in version 3;
            either use ArviZ data structures instead or pass
            ``labels=dataframe.columns`` manually.

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

    title_quantiles : iterable
        A list of 3 fractional quantiles to show as the the upper and lower
        errors. If `None` (default), inherit the values from quantiles, unless
        quantiles is `None`, in which case it defaults to [0.16,0.5,0.84]

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

    axes_scale : str or iterable (ndim,)
        Scale (``"linear"``, ``"log"``) to use for each data dimension. If only
        one scale is specified, use that for all dimensions.

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

    fig : `~matplotlib.figure.Figure`
        Overplot onto the provided figure object, which must either have no
        axes yet, or ``ndim * ndim`` axes already present.  If not set, the
        plot will be drawn on a newly created figure.

    hist_kwargs : dict
        Any extra keyword arguments to send to the 1-D histogram plots.

    **hist2d_kwargs
        Any remaining keyword arguments are sent to :func:`corner.hist2d` to
        generate the 2-D histogram plots.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The ``matplotlib`` figure instance for the corner plot.

    """
    if arviz_corner is None:
        if not (
            isinstance(data, np.ndarray)
            or data.__class__.__name__ == "DataFrame"
        ):
            raise ImportError(
                "Please install arviz or use a numpy array as input"
            )

        if (
            var_names is not None
            or filter_vars is not None
            or coords is not None
            or divergences
            or divergences_kwargs is not None
            or labeller is not None
        ):
            logging.warning(
                "Please install arviz to use the advanced features of corner"
            )

        return corner_impl(
            data,
            bins=bins,
            range=range,
            axes_scale=axes_scale,
            weights=weights,
            color=color,
            hist_bin_factor=hist_bin_factor,
            smooth=smooth,
            smooth1d=smooth1d,
            labels=labels,
            label_kwargs=label_kwargs,
            titles=titles,
            show_titles=show_titles,
            title_quantiles=title_quantiles,
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

    return arviz_corner(
        data,
        bins=bins,
        range=range,
        axes_scale=axes_scale,
        weights=weights,
        color=color,
        hist_bin_factor=hist_bin_factor,
        smooth=smooth,
        smooth1d=smooth1d,
        labels=labels,
        label_kwargs=label_kwargs,
        titles=titles,
        show_titles=show_titles,
        title_quantiles=title_quantiles,
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
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        divergences=divergences,
        divergences_kwargs=divergences_kwargs,
        labeller=labeller,
        **hist2d_kwargs,
    )
