# -*- coding: utf-8 -*-

__all__ = ["arviz_corner"]

import logging
from collections.abc import Mapping

import numpy as np
from arviz.data import convert_to_dataset
from arviz.utils import _var_names, get_coords

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


from .core import corner_impl, overplot_points


def arviz_corner(
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
    fig = corner_impl(
        samples,
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
            overplot_points(
                fig,
                samples[diverging_mask],
                reverse=reverse,
                **divergences_kwargs,
            )

    return fig
