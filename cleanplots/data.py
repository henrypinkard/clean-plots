"""
data.py — Safe pandas helpers for experimental data.

Designed for ML training logs: tidy data with independent variables
(model, lr), nuisance variables (seed), and dependent variables
(train_loss, val_loss) over a time axis (epoch).

Core philosophy: loud errors over silent wrong numbers.
"""

import numpy as np
import pandas as pd


def to_wide(tidy_dataframe, index, values, values_name='metric'):
    """Pivot tidy data to wide format with no silent aggregation.

    Every column not in index or values becomes a column level.
    Uses pivot (not pivot_table), so duplicate entries raise
    ValueError instead of being silently averaged.

    If only one value column is specified, the values level is
    dropped from the MultiIndex for cleaner output.

    Args:
        tidy_dataframe: Tidy DataFrame (one observation per row).
        index: Column(s) to use as row index (e.g. 'epoch').
        values: Column(s) containing dependent variables
                (e.g. ['train_loss', 'val_loss']).
        values_name: Name for the column level created from values.

    Returns:
        DataFrame with MultiIndex columns: (values_name, *remaining_columns).
        If only one value is given, the values level is dropped.
    """
    if isinstance(index, str):
        index = [index]
    if isinstance(values, str):
        values = [values]
    columns = [c for c in tidy_dataframe.columns if c not in index and c not in values]
    result = tidy_dataframe.pivot(index=index, columns=columns, values=values)
    if len(values) == 1:
        result = result.droplevel(0, axis=1)
    else:
        result.columns.names = [values_name] + list(result.columns.names[1:])
    return result


def agg_over(wide_dataframe, col_level, func):
    """Aggregate over a column level, like xarray's .mean(dim=...).

    Moves the specified column level to rows via stack, then
    aggregates over it, preserving all other index and column levels.

    Note on std: use func='std' (pandas default, ddof=1) for sample
    std, not np.std (ddof=0). With few seeds, the difference matters.

    Args:
        wide_dataframe: Wide DataFrame with MultiIndex columns.
        col_level: Name of the column level to aggregate over (e.g. 'seed').
        func: Aggregation function (e.g. 'mean', 'std', np.median, or callable).

    Returns:
        DataFrame with the specified level collapsed.
    """
    stacked = wide_dataframe.stack(col_level)
    remaining = [n for n in stacked.index.names if n != col_level]
    return stacked.groupby(level=remaining).agg(func)


def collapse(tidy_dataframe, index, values, over, func):  # no default — force explicit choice
    """Pivot tidy data and aggregate over a nuisance variable in one step.

    Combines to_wide and agg_over: first pivots losslessly, then
    aggregates over the specified column level.

    Args:
        tidy_dataframe: Tidy DataFrame.
        index: Column(s) for the row index.
        values: Dependent variable column(s).
        over: Column to aggregate over (e.g. 'seed').
        func: Aggregation function or list of functions
              (e.g. 'mean', 'std', ['mean', 'std']).

    Returns:
        Wide DataFrame with the specified variable collapsed.
        If func is a list, the resulting MultiIndex columns get
        a 'stat' level for the aggregation functions.
    """
    single_value = isinstance(values, str)
    if single_value:
        values_list = [values]
    else:
        values_list = list(values)

    wide = to_wide(tidy_dataframe, index=index, values=values)
    result = agg_over(wide, col_level=over, func=func)

    if isinstance(func, list) and isinstance(result.columns, pd.MultiIndex):
        # agg with a list creates an unnamed level for the functions
        names = list(result.columns.names)
        names[-1] = 'stat'  # last level is the one agg added
        result.columns.names = names

    # For single-value collapse, name the result "{func} {value}"
    if single_value and not isinstance(func, list):
        func_name = func if isinstance(func, str) else getattr(func, '__name__', str(func))
        name = f"{func_name} {values_list[0]}"
        if isinstance(result, pd.Series):
            result.name = name
        elif hasattr(result, 'columns'):
            result.columns.name = name

    return result


def slice(wide_dataframe, **kwargs):
    """Select by column level name, like xarray's .sel().

    Values can be a single value or a list of values.

    Args:
        wide_dataframe: DataFrame with MultiIndex columns.
        **kwargs: level_name=value pairs to filter on.

    Returns:
        DataFrame filtered to matching columns.
    """
    for level, value in kwargs.items():
        if isinstance(value, list):
            mask = wide_dataframe.columns.get_level_values(level).isin(value)
            wide_dataframe = wide_dataframe.loc[:, mask]
        else:
            wide_dataframe = wide_dataframe.xs(value, level=level, axis=1)
    return wide_dataframe
