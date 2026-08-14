#convenience code for plot export

from __future__ import annotations

import warnings
from typing import overload, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.projections as mprojections
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.font_manager as fm
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation, LogFormatterMathtext

__all__ = [
    'fig',
    'clean',
    'colors',
    'get_color_cycle',
]

# Default color cycle, in a deliberate colorblind-friendly order -- do not reorder.
# https://davidmathlogic.com/colorblind/#%23179EE8-%2357B50F-%235A00A0-%23419292-%23D900FF-%23F37C2F-%23ACAD9D-%23FF005B-%23E8B70F-%238B4513
colors =  ['#179EE8', '#57B50F', '#5A00A0', '#419292',
           '#D900FF', '#F37C2F', '#ACAD9D', '#FF005B',
           '#E8B70F', '#8B4513',]

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

def _set_style(small_size=SMALL_SIZE, medium_size=MEDIUM_SIZE, bigger_size=BIGGER_SIZE,
              color_cycle=None):
    """Apply the cleanplots matplotlib style.

    Called automatically on import. Can be called again to re-apply or customize.
    """
    #editable text in fonts
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    #make text on figures look good
    plt.rc('font', size=small_size)          # controls default text sizes
    # Use Arial if available, otherwise fall back to DejaVu Sans
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if 'Arial' in available_fonts:
        plt.rcParams['font.sans-serif'] = ['Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams["font.family"] = "sans-serif"

    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', width=2, size=5)
    plt.rc('ytick.major', width=2, size=5)
    plt.rc('xtick.minor', width=1, size=2.5)
    plt.rc('ytick.minor', width=1, size=2.5)

    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size, frameon=False, handletextpad=0.4)    # legend fontsize, no box
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    mpl.rcParams['axes.prop_cycle'] = cycler(color=color_cycle or colors)

# Apply style on import
_set_style()


def _is_dataframe(obj):
    """Check if obj is a pandas DataFrame (duck-typed, no pandas import required)."""
    return hasattr(obj, 'columns') and hasattr(obj, 'index') and hasattr(obj, 'iteritems') or \
           (hasattr(obj, 'columns') and hasattr(obj, 'index') and hasattr(obj, 'iloc'))

def _is_series(obj):
    """Check if obj is a pandas Series (duck-typed, no pandas import required)."""
    return hasattr(obj, 'index') and hasattr(obj, 'values') and hasattr(obj, 'name') and not hasattr(obj, 'columns')

def _split_asymmetric_err(err):
    """Detect Nx2 array error and split into (low, high) tuple.

    Accepts: tuple of (low, high), Nx2 ndarray, or symmetric array.
    Returns the error in a normalized form that downstream code can handle.
    """
    if err is None:
        return err
    if isinstance(err, tuple):
        return err
    arr = np.asarray(err)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0], arr[:, 1])
    return err


class CleanAxes(Axes):
    """Axes subclass with convenience methods for clean plotting."""
    name = 'cleanplots'

    def clean(self, **kwargs):
        """Format this axes. See module-level clean() for full documentation."""
        clean(self, **kwargs)

    def _update_legend(self):
        """Rebuild legend if any labels exist."""
        if getattr(self, '_cleanplots_no_legend', False):
            return
        handles, labels = self.get_legend_handles_labels()
        extra = getattr(self, '_clean_extra_handles', [])
        all_handles = handles + extra
        all_labels = labels + [h.get_label() for h in extra]
        if all_labels:
            super().legend(all_handles, all_labels)

    def line(self, x, y=None, err=None, label=None, err_label=None, err_alpha=0.2, legend=True, log_x=False, log_y=False, **kwargs):
        """Plot a line with optional shaded confidence interval.

        Parameters
        ----------
        x : array-like, list of array-like, Series, or DataFrame
            If a 1-d array-like and y is None, treated as y values with x
            inferred as 0, 1, 2, ... (like matplotlib plot()).
            If a list of array-likes and y is None, each element is plotted
            as a separate line (use label= list for per-line labels). Pass
            a single str to label= to give all lines one shared legend entry
            (handy for plotting many same-colored runs with alpha).
            If a 1-d array-like and y is provided, used as x coordinates.
            If Series, uses index as x and values as y.
            If DataFrame with scalar cells, plots each column as a separate
            line using the index as x and column names as labels.
            If DataFrame with array cells (pivot mode), rows → colors,
            columns → line styles. Legend auto-generated from index and column
            names. Pass y for shared x values or a dict for per-row x values.
        y : array-like, dict, or optional
            For array mode: the y coordinates.
            For pivot DataFrame mode: shared x values (array-like), or a dict
            mapping index values to per-row x arrays. If None, x is inferred.
        err : array-like, list of array-like, tuple of (low, high), Series,
              or DataFrame, optional
            Error data for shaded confidence intervals.
            For multi-line mode: a list of error arrays, one per line.
            For array mode: symmetric error or (low, high) bounds.
            For DataFrame mode: a DataFrame with matching columns.
        label : str or list of str, optional
            Legend label(s). If a list, one label per line (for multi-line mode
            or DataFrame mode). For Series mode, defaults to series.name.
        err_label : str or list of str, optional
            Legend label(s) for the error band.
            If a single str, shown once for all lines (shared label).
            If a list of str, one per line (use None entries to skip).
        err_alpha : float, default 0.2
            Opacity of the shaded error region.
        **kwargs
            Additional keyword arguments passed to plot()
            (e.g. color, linewidth, marker, alpha).
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_x:
            self.set_xscale('log')
        if log_y:
            self.set_yscale('log')
        # DataFrame mode — check both arg positions: line(df) or line(x, df)
        df = None
        x_values = None
        if _is_dataframe(x) and y is None:
            df = x
        elif _is_dataframe(x) and y is not None:
            # Shouldn't happen in normal usage, but handle gracefully
            df = x
            x_values = y
        elif _is_dataframe(y):
            df = y
            x_values = x

        if df is not None:
            # Detect pivot mode: cells contain arrays, not scalars
            first_val = df.iloc[0, 0]
            if isinstance(first_val, (list, np.ndarray)):
                return self._line_pivot(df, x_values=x_values, err=err, err_label=err_label,
                                        err_alpha=err_alpha, **kwargs)

            # MultiIndex columns: top level → line styles, remaining → colors
            if getattr(df.columns, 'nlevels', 1) > 1:
                return self._line_wide_multi(df, err=err, err_label=err_label,
                                             err_alpha=err_alpha, **kwargs)

            # Scalar DataFrame mode: each column is a line, index is x
            all_lines = []
            user_color = kwargs.get('color', None)
            for i, col in enumerate(df.columns):
                col_err = err[col] if (_is_dataframe(err) and col in err.columns) else None
                col_err_label = self._resolve_err_label(err_label, i, len(df.columns))
                # If user passed color= and label=, suppress per-column labels
                col_label = None if (user_color is not None and label is not None) else str(col)
                kw = dict(kwargs)
                # When color is explicit and multiple columns, differentiate by linestyle
                if user_color is not None and len(df.columns) > 1:
                    kw['linestyle'] = self._LINE_STYLES[i % len(self._LINE_STYLES)]
                lines = self._line_array(df.index, df[col].values, err=col_err,
                                         label=col_label, err_label=col_err_label,
                                         err_alpha=err_alpha, **kw)
                all_lines.extend(lines)
            # When color= and label= are both set, add one color entry + column style entries
            if user_color is not None and label is not None:
                self.plot([], [], color=user_color, linestyle='-', label=label)
                if len(df.columns) > 1:
                    existing_labels = {t.get_label() for t in self.get_lines()}
                    for j, col in enumerate(df.columns):
                        linestyle = self._LINE_STYLES[j % len(self._LINE_STYLES)]
                        if str(col) not in existing_labels:
                            self.plot([], [], color='black', linestyle=linestyle, label=str(col))
            if hasattr(df.index, 'name') and df.index.name:
                self.set_xlabel(str(df.index.name))
            self._update_legend()
            return all_lines

        # Series-of-arrays mode: each row is a separate line. Convert to a
        # single-column pivot DataFrame and dispatch to _line_pivot. Handles
        # both line(series) and line(x, series).
        series_y = None
        x_for_series = None
        if _is_series(y) and len(y) > 0 and isinstance(y.iloc[0], (list, np.ndarray)):
            series_y = y
            x_for_series = x
        elif _is_series(x) and y is None and len(x) > 0 and isinstance(x.iloc[0], (list, np.ndarray)):
            series_y = x
        if series_y is not None:
            df_y = series_y.to_frame()
            err_arg = err
            if _is_series(err) and len(err) > 0 and isinstance(err.iloc[0], (list, np.ndarray)):
                err_arg = err.to_frame()
            return self._line_pivot(df_y, x_values=x_for_series, err=err_arg,
                                    err_label=err_label, err_alpha=err_alpha, **kwargs)

        # Series mode: use index as x, values as y
        if _is_series(x):
            series = x
            lbl = label
            err_vals = err.values if _is_series(err) else err
            if hasattr(series.index, 'name') and series.index.name:
                self.set_xlabel(str(series.index.name))
            if series.name is not None:
                self.set_ylabel(str(series.name))
            return self._line_array(series.index, series.values, err=err_vals,
                                    label=lbl, err_label=err_label, err_alpha=err_alpha, **kwargs)

        # Multi-line mode: list of y arrays with no y argument
        if y is None and isinstance(x, (list, tuple)) and len(x) > 0 and not np.isscalar(x[0]):
            y_list = x
            n = len(y_list)
            if isinstance(label, (list, tuple)):
                labels = list(label)
            elif label is None:
                labels = [None] * n
            else:
                # Single string: one shared legend entry, attached to first line only
                labels = [label] + [None] * (n - 1)
            errs = err if isinstance(err, (list, tuple)) and not isinstance(err, tuple) else [err] * n
            # Handle err being a list of arrays vs a tuple (low, high)
            if isinstance(err, list) and len(err) == n and not np.isscalar(err[0]):
                errs = err
            else:
                errs = [err] * n
            all_lines = []
            for i in range(n):
                yi = np.asarray(y_list[i])
                xi = np.arange(len(yi))
                lbl = labels[i] if i < len(labels) else None
                ei = errs[i]
                el = self._resolve_err_label(err_label, i, n)
                lines = self._line_array(xi, yi, err=ei, label=lbl,
                                         err_label=el, err_alpha=err_alpha, **kwargs)
                all_lines.extend(lines)
            self._update_legend()
            return all_lines

        # Single-arg mode: y values only, infer x
        if y is None:
            y = np.asarray(x)
            x = np.arange(len(y))

        # Array mode
        return self._line_array(x, y, err=err, label=label, err_label=err_label,
                                err_alpha=err_alpha, **kwargs)

    @staticmethod
    def _resolve_err_label(err_label, index, total):
        """Determine the err_label for the i-th line in a multi-line call.

        If err_label is a list, return the i-th entry.
        If err_label is a single string (shared), return it only for the first line.
        """
        if err_label is None:
            return None
        if isinstance(err_label, (list, tuple)):
            return err_label[index] if index < len(err_label) else None
        # Single string: show only on first line
        return err_label if index == 0 else None

    def _line_array(self, x, y, err=None, label=None, err_label=None, err_alpha=0.2, **kwargs):
        """Plot a single line with optional shaded confidence interval (array data)."""
        lines = self.plot(x, y, label=label, **kwargs)
        color = lines[0].get_color()
        if err is not None:
            err = _split_asymmetric_err(err)
            if isinstance(err, tuple):
                low, high = np.asarray(err[0]), np.asarray(err[1])
            else:
                y = np.asarray(y)
                err = np.asarray(err)
                low, high = y - err, y + err
            self.fill_between(x, low, high, alpha=err_alpha, color=color,
                              linewidth=0, label='_nolegend_')
            if err_label is not None:
                band_handle = mpatches.Patch(facecolor='gray', alpha=err_alpha, label=err_label)
                band_handle._cleanplots_skip_color_label = True
                if not hasattr(self, '_clean_extra_handles'):
                    self._clean_extra_handles = []
                self._clean_extra_handles.append(band_handle)
        self._update_legend()
        return lines

    _LINE_STYLES = ['-', '--', ':', '-.']

    def _line_pivot(self, df, x_values=None, err=None, err_label=None, err_alpha=0.2, **kwargs):
        """Plot from a pivot DataFrame where each cell contains an array.

        Rows (index) → colors, columns → line styles.

        Parameters
        ----------
        df : DataFrame
            Pivot table where each cell is a list/array of y values.
        x_values : array-like, dict, or None
            Shared x values, or a dict mapping index values to per-row x arrays.
            If None, inferred as range(len) from the first cell.
        err : DataFrame or None
            Same shape as df, where each cell is an error array.
        err_label : str or None
            Shared label for error bands (shown once).
        err_alpha : float
            Opacity of error bands.
        **kwargs
            Passed to plot().
        """
        cycle = get_color_cycle()
        all_lines = []
        first_err = True
        user_color = kwargs.pop('color', None)
        user_label = kwargs.pop('label', None)

        for i, row_key in enumerate(df.index):
            color = user_color if user_color is not None else cycle[i % len(cycle)]
            for j, col in enumerate(df.columns):
                y_data = np.asarray(df.loc[row_key, col])
                linestyle = self._LINE_STYLES[j % len(self._LINE_STYLES)]

                # Resolve x values
                if x_values is None:
                    x_data = np.arange(len(y_data))
                elif isinstance(x_values, dict):
                    x_data = np.asarray(x_values[row_key])
                else:
                    x_data = np.asarray(x_values)

                # Resolve error
                cell_err = None
                if _is_dataframe(err):
                    cell_err_raw = err.loc[row_key, col]
                    if cell_err_raw is not None:
                        cell_err = np.asarray(cell_err_raw)

                # err_label only on first line that has error
                el = None
                if cell_err is not None and err_label is not None and first_err:
                    el = err_label
                    first_err = False

                kw = dict(kwargs)
                kw['color'] = color
                kw['linestyle'] = linestyle
                lines = self._line_array(x_data, y_data, err=cell_err,
                                         err_label=el, err_alpha=err_alpha, **kw)
                all_lines.extend(lines)

        # Legend: color entries for index
        if user_color is not None and user_label is not None:
            # Single color with explicit label: one color legend entry
            self.plot([], [], color=user_color, linestyle='-', label=user_label)
        elif user_color is None:
            for i, row_key in enumerate(df.index):
                color = cycle[i % len(cycle)]
                self.plot([], [], color=color, linestyle='-', label=str(row_key))

        # Single column: use column name as ylabel instead of legend entry
        if len(df.columns) == 1:
            self.set_ylabel(str(df.columns[0]))
        else:
            # Multiple columns: add linestyle legend entries (always black, deduplicated)
            existing_labels = {t.get_label() for t in self.get_lines()}
            for j, col in enumerate(df.columns):
                linestyle = self._LINE_STYLES[j % len(self._LINE_STYLES)]
                if str(col) not in existing_labels:
                    self.plot([], [], color='black', linestyle=linestyle, label=str(col))

        self._update_legend()
        return all_lines

    def _line_wide_multi(self, df, err=None, err_label=None, err_alpha=0.2, **kwargs):
        """Plot a scalar-cell DataFrame with MultiIndex columns.

        Top column level → line styles, remaining levels → colors.
        If color= is passed explicitly, all lines use that color
        (no per-group color legend).
        """
        cycle = get_color_cycle()
        all_lines = []
        first_err = True
        user_color = kwargs.pop('color', None)
        user_label = kwargs.pop('label', None)

        top_values = df.columns.get_level_values(0).unique()

        # Group keys from remaining column levels
        sub0 = df.xs(top_values[0], level=0, axis=1)
        group_keys = sub0.columns.tolist()

        for i, gk in enumerate(group_keys):
            color = user_color if user_color is not None else cycle[i % len(cycle)]
            for j, tv in enumerate(top_values):
                linestyle = self._LINE_STYLES[j % len(self._LINE_STYLES)]
                sub = df.xs(tv, level=0, axis=1)
                y_data = sub[gk].values

                cell_err = None
                if _is_dataframe(err):
                    err_sub = err.xs(tv, level=0, axis=1)
                    cell_err = err_sub[gk].values

                el = None
                if cell_err is not None and err_label is not None and first_err:
                    el = err_label
                    first_err = False

                kw = dict(kwargs)
                kw['color'] = color
                kw['linestyle'] = linestyle
                lines = self._line_array(df.index, y_data, err=cell_err,
                                         err_label=el, err_alpha=err_alpha, **kw)
                all_lines.extend(lines)

        # Legend: color entries for groups
        if user_color is not None and user_label is not None:
            # Single color with explicit label: one color legend entry
            self.plot([], [], color=user_color, linestyle='-', label=user_label)
        elif user_color is None:
            # Auto-colored groups: one entry per group
            for i, gk in enumerate(group_keys):
                color = cycle[i % len(cycle)]
                if isinstance(gk, tuple):
                    label_str = ', '.join(str(v) for v in gk)
                else:
                    label_str = str(gk)
                self.plot([], [], color=color, linestyle='-', label=label_str)

        # Line style entries for top level (always black, deduplicated)
        if len(top_values) > 1:
            existing_labels = {t.get_label() for t in self.get_lines()}
            for j, tv in enumerate(top_values):
                linestyle = self._LINE_STYLES[j % len(self._LINE_STYLES)]
                if str(tv) not in existing_labels:
                    self.plot([], [], color='black', linestyle=linestyle, label=str(tv))
        else:
            self.set_ylabel(str(top_values[0]))

        if hasattr(df.index, 'name') and df.index.name:
            self.set_xlabel(str(df.index.name))

        self._update_legend()
        return all_lines

    def bar(self, x, height=None, err=None, err_label=None, capsize=4, err_kw=None, legend=True, log_y=False, **kwargs):
        """Plot a bar chart with optional error bars.

        Parameters
        ----------
        x : array-like, Series, or DataFrame
            If numeric array-like and height is None, treated as bar heights
            with x positions inferred as 0, 1, 2, ...
            If array-like and height is provided, used as bar positions/labels.
            If Series, uses index as labels and values as heights.
            If DataFrame, plots grouped bars with one group per column, using
            the index as labels and column names as legend entries.
        height : array-like, optional
            Bar heights. If None and x is numeric, x is used as heights.
        err : array-like, tuple of (low, high), Series, or DataFrame, optional
            Error bar data. For DataFrame mode, a DataFrame with matching
            columns provides per-group error bars.
        err_label : str, optional
            Legend label for the error bars (e.g. '±1 SD').
            In DataFrame mode, automatically shown only once.
        capsize : float, default 4
            Width of error bar caps in points.
        err_kw : dict, optional
            Additional keyword arguments for the error bars.
        **kwargs
            Additional keyword arguments passed to matplotlib bar().
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_y:
            self.set_yscale('log')
        # DataFrame mode: grouped bars
        if _is_dataframe(x):
            df = x
            n_groups = len(df.columns)
            x_pos = np.arange(len(df.index))
            width = 0.8 / n_groups
            containers = []
            for i, col in enumerate(df.columns):
                col_err = err[col].values if (_is_dataframe(err) and col in err.columns) else None
                col_err_label = err_label if (i == 0 and err_label is not None) else None
                offset = (i - (n_groups - 1) / 2) * width
                container = self._bar_array(x_pos + offset, df[col].values, err=col_err,
                                            err_label=col_err_label, capsize=capsize,
                                            err_kw=err_kw, width=width, label=str(col), **kwargs)
                containers.append(container)
            self.set_xticks(x_pos)
            self.set_xticklabels([str(l) for l in df.index])
            if hasattr(df.index, 'name') and df.index.name:
                self.set_xlabel(str(df.index.name))
            self._update_legend()
            return containers

        # Series mode
        if _is_series(x):
            series = x
            lbl = kwargs.pop('label', None)
            err_vals = err.values if _is_series(err) else err
            if hasattr(series.index, 'name') and series.index.name:
                self.set_xlabel(str(series.index.name))
            if series.name is not None:
                self.set_ylabel(str(series.name))
            return self._bar_array([str(l) for l in series.index], series.values,
                                   err=err_vals, err_label=err_label, capsize=capsize,
                                   err_kw=err_kw, label=lbl, **kwargs)

        # Single-argument mode: numeric array as heights, infer x
        if height is None:
            height = np.asarray(x)
            x = np.arange(len(height))

        # Array mode
        return self._bar_array(x, height, err=err, err_label=err_label,
                               capsize=capsize, err_kw=err_kw, **kwargs)

    def _bar_array(self, x, height, err=None, err_label=None, capsize=4, err_kw=None, **kwargs):
        """Plot bars from array data."""
        if err_kw is None:
            err_kw = {}
        err_kw.setdefault('color', 'black')
        err_kw.setdefault('linewidth', 1.5)
        err = _split_asymmetric_err(err)
        if isinstance(err, tuple):
            err = np.array([np.asarray(err[0]), np.asarray(err[1])])
        container = super().bar(x, height, yerr=err, capsize=capsize,
                                error_kw=err_kw, **kwargs)
        if err is not None and err_label is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.errorbar([np.nan], [np.nan], yerr=[0.5], fmt='none',
                              color=err_kw['color'], capsize=capsize,
                              linewidth=err_kw['linewidth'], label=err_label)
        self._update_legend()
        return container

    def barh(self, y, width=None, err=None, err_label=None, capsize=4, err_kw=None,
             legend=True, log_x=False, **kwargs):
        """Horizontal bar chart with optional error bars.

        Parameters
        ----------
        y : array-like, Series, or DataFrame
            If numeric array-like and width is None, treated as bar widths
            with y positions inferred as 0, 1, 2, ...
            If array-like and width is provided, used as bar positions/labels.
            If Series, uses index as labels and values as widths.
            If DataFrame, plots grouped bars with one group per column, using
            the index as labels and column names as legend entries.
        width : array-like, optional
            Bar widths. If None and y is numeric, y is used as widths.
        err : array-like, tuple of (low, high), Series, or DataFrame, optional
            Error bar data.
        err_label : str, optional
            Legend label for the error bars.
        capsize : float, default 4
            Width of error bar caps in points.
        err_kw : dict, optional
            Additional keyword arguments for the error bars.
        **kwargs
            Additional keyword arguments passed to matplotlib barh().
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_x:
            self.set_xscale('log')
        # DataFrame mode: grouped bars
        if _is_dataframe(y):
            df = y
            n_groups = len(df.columns)
            y_pos = np.arange(len(df.index))
            height = 0.8 / n_groups
            containers = []
            for i, col in enumerate(df.columns):
                col_err = err[col].values if (_is_dataframe(err) and col in err.columns) else None
                col_err_label = err_label if (i == 0 and err_label is not None) else None
                offset = (i - (n_groups - 1) / 2) * height
                container = self._barh_array(y_pos + offset, df[col].values, err=col_err,
                                             err_label=col_err_label, capsize=capsize,
                                             err_kw=err_kw, height=height, label=str(col), **kwargs)
                containers.append(container)
            self.set_yticks(y_pos)
            self.set_yticklabels([str(l) for l in df.index])
            if hasattr(df.index, 'name') and df.index.name:
                self.set_ylabel(str(df.index.name))
            self._update_legend()
            return containers

        # Series mode
        if _is_series(y):
            series = y
            lbl = kwargs.pop('label', None)
            err_vals = err.values if _is_series(err) else err
            if hasattr(series.index, 'name') and series.index.name:
                self.set_ylabel(str(series.index.name))
            if series.name is not None:
                self.set_xlabel(str(series.name))
            return self._barh_array([str(l) for l in series.index], series.values,
                                    err=err_vals, err_label=err_label, capsize=capsize,
                                    err_kw=err_kw, label=lbl, **kwargs)

        # Single-argument mode: numeric array as widths, infer y
        if width is None:
            width = np.asarray(y)
            y = np.arange(len(width))

        # Array mode
        return self._barh_array(y, width, err=err, err_label=err_label,
                                capsize=capsize, err_kw=err_kw, **kwargs)

    def _barh_array(self, y, width, err=None, err_label=None, capsize=4, err_kw=None, **kwargs):
        """Plot horizontal bars from array data."""
        if err_kw is None:
            err_kw = {}
        err_kw.setdefault('color', 'black')
        err_kw.setdefault('linewidth', 1.5)
        err = _split_asymmetric_err(err)
        if isinstance(err, tuple):
            err = np.array([np.asarray(err[0]), np.asarray(err[1])])
        # Call Axes.bar directly with orientation='horizontal' to bypass
        # our bar() override (matplotlib's barh calls self.bar internally)
        height = kwargs.pop('height', 0.8)
        container = Axes.bar(self, x=None, height=height, width=width, bottom=y,
                             align='center', orientation='horizontal',
                             xerr=err, capsize=capsize,
                             error_kw=err_kw, **kwargs)
        if err is not None and err_label is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.errorbar([np.nan], [np.nan], xerr=[0.5], fmt='none',
                              color=err_kw['color'], capsize=capsize,
                              linewidth=err_kw['linewidth'], label=err_label)
        self._update_legend()
        return container

    def scatter(self, x, y=None, xerr=None, yerr=None, err_label=None,
                xerr_label=None, yerr_label=None, capsize=4, err_kw=None,
                label=None, group=None, legend=True, log_x=False, log_y=False, **kwargs):
        """Scatter plot with optional error bars in x and/or y.

        Parameters
        ----------
        x : array-like (Nx2 or Nx3), DataFrame, or 1-d array-like
            If a 2-d array with 2 columns, columns are used as (x, y).
            If a 2-d array with 3 columns, columns are (x, y, size).
            If a DataFrame with 2+ columns, first two are x and y;
            column names become axis labels.
            Otherwise, 1-d x coordinates (y must be provided).
        y : array-like, optional
            The y coordinates. Required when x is 1-d.
        xerr : array-like or tuple of (low, high), optional
            Horizontal error bars. Symmetric if array-like, asymmetric if tuple.
        yerr : array-like or tuple of (low, high), optional
            Vertical error bars. Symmetric if array-like, asymmetric if tuple.
        err_label : str, optional
            Legend label for error bars when both directions share a label
            (or as a fallback when only one direction is present).
        xerr_label : str, optional
            Legend label for horizontal error bars.
        yerr_label : str, optional
            Legend label for vertical error bars.
        capsize : float, default 4
            Width of error bar caps in points.
        err_kw : dict, optional
            Additional keyword arguments for the error bars.
        label : str, optional
            Legend label for the scatter points.
        group : array-like, optional
            Categorical group labels. Each unique value gets a color from
            the cycle and a legend entry. Mutually exclusive with 'c'.
        **kwargs
            Additional keyword arguments passed to matplotlib scatter()
            (e.g. color, s, marker).
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_x:
            self.set_xscale('log')
        if log_y:
            self.set_yscale('log')
        # Matrix / DataFrame mode: Nx2 or Nx3
        if y is None:
            if _is_dataframe(x):
                col_names = list(x.columns)
                data = x.values
                if len(col_names) >= 1:
                    self.set_xlabel(str(col_names[0]))
                if len(col_names) >= 2:
                    self.set_ylabel(str(col_names[1]))
            else:
                data = np.asarray(x)
            if data.ndim == 2 and data.shape[1] in (2, 3):
                x = data[:, 0]
                y = data[:, 1]
                if data.shape[1] == 3:
                    kwargs.setdefault('s', data[:, 2])
            else:
                raise ValueError("When y is not provided, x must be an Nx2 or Nx3 array or DataFrame")

        # Group mode: categorical coloring
        if group is not None:
            if 'c' in kwargs:
                raise ValueError("Cannot use both 'group' and 'c'")
            group_arr = np.asarray(group)
            unique_groups = list(dict.fromkeys(group_arr))
            cycle = get_color_cycle()
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            n = len(x_arr)
            all_sc = []
            for i, g in enumerate(unique_groups):
                mask = group_arr == g
                color = cycle[i % len(cycle)]
                masked_kwargs = {}
                for k, v in kwargs.items():
                    if hasattr(v, '__len__') and not isinstance(v, str) and len(v) == n:
                        masked_kwargs[k] = np.asarray(v)[mask]
                    else:
                        masked_kwargs[k] = v
                sc = super().scatter(x_arr[mask], y_arr[mask],
                                     color=color, label=str(g), **masked_kwargs)
                all_sc.append(sc)
            self._update_legend()
            return all_sc

        sc = super().scatter(x, y, label=label, **kwargs)
        if xerr is not None or yerr is not None:
            if err_kw is None:
                err_kw = {}
            err_kw.setdefault('fmt', 'none')
            err_kw.setdefault('color', 'black')
            err_kw.setdefault('linewidth', 1.5)
            err_kw.setdefault('capsize', capsize)
            xerr = _split_asymmetric_err(xerr)
            if isinstance(xerr, tuple):
                xerr = np.array([np.asarray(xerr[0]), np.asarray(xerr[1])])
            yerr = _split_asymmetric_err(yerr)
            if isinstance(yerr, tuple):
                yerr = np.array([np.asarray(yerr[0]), np.asarray(yerr[1])])
            self.errorbar(x, y, xerr=xerr, yerr=yerr, **err_kw)

            # Resolve labels: err_label is a fallback for whichever direction is present
            if xerr_label is None and yerr_label is None and err_label is not None:
                if yerr is not None:
                    yerr_label = err_label
                if xerr is not None and yerr is None:
                    xerr_label = err_label

            # Draw legend dummies with appropriate orientation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if yerr_label is not None and yerr is not None:
                    self.errorbar([np.nan], [np.nan], yerr=[0.5], fmt='none',
                                  color=err_kw['color'], capsize=capsize,
                                  linewidth=err_kw['linewidth'], label=yerr_label)
                if xerr_label is not None and xerr is not None:
                    self.errorbar([np.nan], [np.nan], xerr=[0.5], fmt='none',
                                  color=err_kw['color'], capsize=capsize,
                                  linewidth=err_kw['linewidth'], label=xerr_label)
        self._update_legend()
        return sc

    def add_colorbar(self, cmap, values=None, vmin=None, vmax=None):
        """Add a colorbar to this axes for manually-applied colors.

        Use when you've applied colors via a colormap manually (e.g.
        edgecolors=cmap(values)) and want a colorbar showing the scale.

        Parameters
        ----------
        cmap : Colormap
            The colormap used to generate the colors.
        values : array-like, optional
            The data values that were mapped. Used to infer vmin/vmax.
        vmin, vmax : float, optional
            Explicit range for the colorbar. Override values-based inference.
            Required if values is not provided.

        Returns
        -------
        Colorbar
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if values is not None:
            values = np.asarray(values)
            if vmin is None:
                vmin = float(values.min())
            if vmax is None:
                vmax = float(values.max())
        elif vmin is None or vmax is None:
            raise ValueError("Must provide either 'values' or both 'vmin' and 'vmax'")

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        return self.get_figure().colorbar(sm, ax=self)

    def legend(self, *args, **kwargs):
        """Override legend to include extra handles from line()/bar()/scatter() err_label."""
        extra = getattr(self, '_clean_extra_handles', [])
        if extra:
            existing_handles, existing_labels = self.get_legend_handles_labels()
            handles = existing_handles + extra
            labels = existing_labels + [h.get_label() for h in extra]
            return super().legend(handles, labels, *args, **kwargs)
        return super().legend(*args, **kwargs)

    def color_labels(self, labels, colors=None, x=0.95, y=0.95, fontsize=None, **kwargs):
        """Place colored text labels on the axes (legend alternative).

        Each label is displayed in its corresponding color, stacked vertically.
        Useful for labeling lines directly instead of using a legend box.

        Parameters
        ----------
        labels : str or list of str
            Label text(s) to display.
        colors : color or list of colors, optional
            Color(s) for each label. Defaults to the current color cycle.
        x : float, default 0.95
            Horizontal position in axes coordinates (0=left, 1=right).
        y : float, default 0.95
            Vertical position of the first (top) label in axes coordinates.
        fontsize : float, optional
            Font size in points. Defaults to matplotlib's current font size.
        **kwargs
            Additional keyword arguments passed to ax.text()
            (e.g. fontweight, fontstyle, ha).
        """
        if isinstance(labels, str):
            labels = [labels]
        if colors is None:
            colors = get_color_cycle()
        if isinstance(colors, str):
            colors = [colors]

        if fontsize is None:
            fontsize = plt.rcParams['font.size']

        # Compute line spacing: convert font points to axes-fraction
        fig = self.get_figure()
        bbox = self.get_position()
        fig_height_pts = fig.get_size_inches()[1] * 72
        ax_height_pts = bbox.height * fig_height_pts
        line_spacing = fontsize * 1.5 / ax_height_pts

        kwargs.setdefault('ha', 'right')
        kwargs.setdefault('va', 'top')

        texts = []
        for i, label in enumerate(labels):
            color = colors[i % len(colors)]
            t = self.text(x, y - i * line_spacing, label, color=color,
                          transform=self.transAxes, fontsize=fontsize, **kwargs)
            texts.append(t)
        return texts

    def heatmap(self, data, annotate=True, fmt='.2f', annot_strings=None,
                cmap='viridis', cbar=True, legend=True, **kwargs):
        """Plot a heatmap from a 2-d array or DataFrame.

        Parameters
        ----------
        data : 2-d array-like or DataFrame
            The data to display. If a DataFrame, index → y-axis labels,
            columns → x-axis labels. Index and column names become
            axis labels automatically.
        annotate : bool, default True
            Whether to annotate each cell with its value.
        fmt : str, default '.2f'
            Format string for cell annotations (e.g. '.2f', 'd', '.1%').
            Ignored if annot_strings is provided.
        annot_strings : 2-d array-like or DataFrame, optional
            Custom annotation strings for each cell. Must have the same
            shape as data. If provided, these are used instead of
            formatting the numeric values with fmt.
        cmap : str or Colormap, default 'viridis'
            Colormap for the heatmap.
        cbar : bool, default True
            Whether to add a colorbar.
        **kwargs
            Additional keyword arguments passed to imshow().
        """
        if not legend:
            self._cleanplots_no_legend = True
        row_labels = None
        col_labels = None
        row_name = None
        col_name = None
        if _is_dataframe(data):
            row_labels = [str(l) for l in data.index]
            col_labels = [str(l) for l in data.columns]
            row_name = getattr(data.index, 'name', None)
            col_name = getattr(data.columns, 'name', None)
            values = data.values.astype(float)
        else:
            values = np.asarray(data, dtype=float)

        kwargs.setdefault('aspect', 'auto')
        im = self.imshow(values, cmap=cmap, **kwargs)

        if row_labels is not None:
            self.set_yticks(range(len(row_labels)))
            self.set_yticklabels(row_labels)
        if col_labels is not None:
            self.set_xticks(range(len(col_labels)))
            self.set_xticklabels(col_labels)
            self.xaxis.set_label_position('top')
            self.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        if row_name:
            self.set_ylabel(str(row_name))
        if col_name:
            self.set_xlabel(str(col_name))

        if annot_strings is not None:
            if _is_dataframe(annot_strings):
                annot_strings = annot_strings.values
            annot_strings = np.asarray(annot_strings)
            annotate = True

        if annotate:
            # Use colormap luminance to pick text color for contrast
            norm = im.norm
            cmap_obj = im.cmap
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    rgba = cmap_obj(norm(values[i, j]))
                    # Perceived luminance (ITU-R BT.601)
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    color = 'white' if lum < 0.5 else 'black'
                    if annot_strings is not None:
                        txt = str(annot_strings[i, j])
                    else:
                        val = int(values[i, j]) if fmt == 'd' else values[i, j]
                        txt = format(val, fmt)
                    self.text(j, i, txt,
                              ha='center', va='center', color=color)

        if cbar:
            return self.get_figure().colorbar(im, ax=self)
        return None

    def box(self, data, positions=None, widths=0.6, showfliers=True,
            patch_artist=True, color=None, legend=True, log_y=False, **kwargs):
        """Box plot from a DataFrame, dict, or list of arrays.

        Parameters
        ----------
        data : DataFrame, dict, or list of array-like
            If DataFrame with scalar columns: each column is a box.
            If DataFrame with array cells (pivot mode): rows → colors,
            columns → grouped box positions. Index names label the colors,
            column names label the x positions.
            If dict: keys are labels, values are arrays.
            If list: each element is an array for one box.
        positions : array-like, optional
            Custom x positions for boxes.
        widths : float, default 0.6
            Width of each box.
        showfliers : bool, default True
            Whether to show outlier points.
        patch_artist : bool, default True
            Use filled boxes (enables color).
        color : str, list, or None, default None
            Color for the boxes. If None, all boxes use the first color in
            the cycle. If a string, all boxes use that color. If a list,
            each box is colored by the corresponding entry.
        **kwargs
            Additional keyword arguments passed to matplotlib boxplot().
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_y:
            self.set_yscale('log')
        cycle = get_color_cycle()

        # DataFrame pivot mode: cells contain arrays
        if _is_dataframe(data):
            first_val = data.iloc[0, 0]
            if isinstance(first_val, (list, np.ndarray)):
                return self._box_pivot(data, widths=widths, showfliers=showfliers,
                                       patch_artist=patch_artist, **kwargs)
            box_data = [data[col].dropna().values for col in data.columns]
            labels = [str(c) for c in data.columns]
        elif isinstance(data, dict):
            box_data = [np.asarray(v) for v in data.values()]
            labels = [str(k) for k in data.keys()]
        else:
            box_data = data
            labels = None

        n = len(box_data)
        if color is None:
            box_colors = [cycle[0]] * n
        elif isinstance(color, (list, tuple, np.ndarray)):
            box_colors = [color[i % len(color)] for i in range(n)]
        else:
            box_colors = [color] * n

        bp_kwargs = dict(widths=widths, showfliers=showfliers, patch_artist=patch_artist)
        if labels is not None:
            bp_kwargs['tick_labels'] = labels
        if positions is not None:
            bp_kwargs['positions'] = positions
        bp_kwargs.update(kwargs)

        bp = self.boxplot(box_data, **bp_kwargs)

        if patch_artist:
            for patch, c in zip(bp['boxes'], box_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
        for median in bp.get('medians', []):
            median.set_color('black')
        for mean in bp.get('means', []):
            mean.set_color('black')
        return bp

    def _box_pivot(self, df, widths=0.6, showfliers=True, patch_artist=True, **kwargs):
        """Grouped box plot from a pivot DataFrame where cells contain arrays.

        Index → x-axis labels, columns → colored groups (matches bar chart convention).
        """
        cycle = get_color_cycle()
        n_rows = len(df.index)
        n_cols = len(df.columns)
        box_width = widths / n_cols

        all_box_data = []
        all_positions = []
        all_colors = []

        for i, row_key in enumerate(df.index):
            for j, col in enumerate(df.columns):
                arr = np.asarray(df.loc[row_key, col])
                all_box_data.append(arr)
                offset = (j - (n_cols - 1) / 2) * box_width
                all_positions.append(i + offset)
                all_colors.append(cycle[j % len(cycle)])

        bp = self.boxplot(all_box_data, positions=all_positions, widths=box_width * 0.9,
                          showfliers=showfliers, patch_artist=patch_artist, **kwargs)

        if patch_artist:
            for patch, color in zip(bp['boxes'], all_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        for median in bp.get('medians', []):
            median.set_color('black')
        for mean in bp.get('means', []):
            mean.set_color('black')

        # Set x labels to index names
        self.set_xticks(range(n_rows))
        self.set_xticklabels([str(k) for k in df.index])

        # Single column: ylabel instead of legend
        if n_cols == 1:
            self.set_ylabel(str(df.columns[0]))

        # Legend for column colors
        for j, col in enumerate(df.columns):
            color = cycle[j % len(cycle)]
            self.plot([], [], color=color, linewidth=6, alpha=0.7, label=str(col))
        self._update_legend()

        return bp

    def violin(self, data, positions=None, widths=0.7, showmedians=True,
               showextrema=False, color=None, legend=True, log_y=False, **kwargs):
        """Violin plot from a DataFrame, dict, or list of arrays.

        Parameters
        ----------
        data : DataFrame, dict, or list of array-like
            Same formats as box(). DataFrame with array cells uses pivot mode
            (rows → colors, columns → x positions).
        positions : array-like, optional
            Custom x positions for violins.
        widths : float, default 0.7
            Width of each violin.
        showmedians : bool, default True
            Whether to show median lines.
        showextrema : bool, default False
            Whether to show min/max lines.
        color : str, list, or None, default None
            Color for the violin bodies. If None, all violins use the first
            color in the cycle. If a string, all violins use that color. If
            a list, each violin is colored by the corresponding entry.
        **kwargs
            Additional keyword arguments passed to matplotlib violinplot().
            Useful extras: bw_method=(float | 'scott' | 'silverman') controls
            the KDE bandwidth (default 'scott'); larger values → smoother.
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_y:
            self.set_yscale('log')
        # matplotlib's violinplot() internally calls self.violin(vpstats, ...)
        # where vpstats is a list of dicts. Detect that and delegate to parent.
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return super().violin(data, positions=positions, widths=widths,
                                  showmedians=showmedians, showextrema=showextrema, **kwargs)

        cycle = get_color_cycle()

        # DataFrame pivot mode
        if _is_dataframe(data):
            first_val = data.iloc[0, 0]
            if isinstance(first_val, (list, np.ndarray)):
                return self._violin_pivot(data, widths=widths, showmedians=showmedians,
                                          showextrema=showextrema, **kwargs)
            violin_data = [data[col].dropna().values for col in data.columns]
            labels = [str(c) for c in data.columns]
        elif isinstance(data, dict):
            violin_data = [np.asarray(v) for v in data.values()]
            labels = [str(k) for k in data.keys()]
        else:
            violin_data = [np.asarray(v) for v in data]
            labels = None

        n = len(violin_data)
        if color is None:
            body_colors = [cycle[0]] * n
        elif isinstance(color, (list, tuple, np.ndarray)):
            body_colors = [color[i % len(color)] for i in range(n)]
        else:
            body_colors = [color] * n

        vp = self.violinplot(violin_data,
                             positions=positions if positions is not None else range(n),
                             widths=widths, showmedians=showmedians,
                             showextrema=showextrema, **kwargs)
        for body, c in zip(vp['bodies'], body_colors):
            body.set_facecolor(c)
            body.set_alpha(0.7)
        for key in ('cmedians', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
            if key in vp:
                vp[key].set_color('black')
        if labels is not None:
            self.set_xticks(range(n))
            self.set_xticklabels(labels)
        return vp

    def _violin_pivot(self, df, widths=0.7, showmedians=True, showextrema=False, **kwargs):
        """Grouped violin plot from a pivot DataFrame where cells contain arrays.

        Index → x-axis labels, columns → colored groups (matches bar chart convention).
        """
        cycle = get_color_cycle()
        n_rows = len(df.index)
        n_cols = len(df.columns)
        violin_width = widths / n_cols

        all_data = []
        all_positions = []
        all_colors = []

        for i, row_key in enumerate(df.index):
            for j, col in enumerate(df.columns):
                arr = np.asarray(df.loc[row_key, col])
                all_data.append(arr)
                offset = (j - (n_cols - 1) / 2) * violin_width
                all_positions.append(i + offset)
                all_colors.append(cycle[j % len(cycle)])

        vp = self.violinplot(all_data, positions=all_positions, widths=violin_width * 0.9,
                             showmedians=showmedians, showextrema=showextrema, **kwargs)

        for body, color in zip(vp['bodies'], all_colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        for key in ('cmedians', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
            if key in vp:
                vp[key].set_color('black')

        # Set x labels to index names
        self.set_xticks(range(n_rows))
        self.set_xticklabels([str(k) for k in df.index])

        # Single column: ylabel instead of legend
        if n_cols == 1:
            self.set_ylabel(str(df.columns[0]))

        # Legend for column colors
        for j, col in enumerate(df.columns):
            color = cycle[j % len(cycle)]
            self.plot([], [], color=color, linewidth=6, alpha=0.7, label=str(col))
        self._update_legend()

        return vp

    def strip(self, data, positions=None, jitter=0.1, color=None,
              s=25, alpha=0.7, legend=True, log_y=False, **kwargs):
        """Strip (jitter) plot: individual data points at jittered x positions.

        An alternative to box/violin that shows every raw data point. Good
        for small samples where distribution shape is better communicated by
        the points themselves than by an aggregate.

        Parameters
        ----------
        data : DataFrame, dict, or list of array-like
            If DataFrame with array cells (pivot mode): rows → colors,
            columns → x-axis groups (side-by-side strips, analogous to
            grouped bar charts). Index names label the colors, column
            names label the x positions.
            If DataFrame with scalar columns: each column is a strip.
            If dict: keys are labels, values are arrays.
            If list: each element is an array for one strip.
        positions : array-like, optional
            Custom x positions for the groups. Defaults to 0, 1, 2, ...
        jitter : float, default 0.1
            Half-width of the uniform jitter applied to x positions.
        color : str, list, or None, default None
            Same semantics as box() / violin().
        s : float, default 25
            Marker size.
        alpha : float, default 0.7
            Marker alpha.
        **kwargs
            Additional keyword arguments passed to matplotlib scatter().
        """
        if not legend:
            self._cleanplots_no_legend = True
        if log_y:
            self.set_yscale('log')
        cycle = get_color_cycle()

        # DataFrame pivot mode: cells contain arrays
        if _is_dataframe(data):
            first_val = data.iloc[0, 0]
            if isinstance(first_val, (list, np.ndarray)):
                return self._strip_pivot(data, jitter=jitter, s=s, alpha=alpha, **kwargs)

        if _is_dataframe(data):
            groups = [np.asarray(data[col].dropna().values) for col in data.columns]
            labels = [str(c) for c in data.columns]
        elif isinstance(data, dict):
            groups = [np.asarray(v) for v in data.values()]
            labels = [str(k) for k in data.keys()]
        else:
            groups = [np.asarray(g) for g in data]
            labels = None

        n = len(groups)
        if positions is None:
            positions = np.arange(n)
        else:
            positions = np.asarray(positions)

        if color is None:
            group_colors = [cycle[0]] * n
        elif isinstance(color, (list, tuple, np.ndarray)):
            group_colors = [color[i % len(color)] for i in range(n)]
        else:
            group_colors = [color] * n

        rng = np.random.default_rng(0)
        for pos, group, c in zip(positions, groups, group_colors):
            x = pos + rng.uniform(-jitter, jitter, size=len(group))
            Axes.scatter(self, x, group, s=s, alpha=alpha, color=c, **kwargs)

        if labels is not None:
            self.set_xticks(list(positions))
            self.set_xticklabels(labels)

    def _strip_pivot(self, df, jitter=0.1, s=25, alpha=0.7, **kwargs):
        """Grouped strip plot from a pivot DataFrame where cells contain arrays.

        Index → x-axis labels, columns → colored groups (matches bar chart convention).
        """
        cycle = get_color_cycle()
        n_rows = len(df.index)
        n_cols = len(df.columns)
        strip_width = 0.6 / n_cols

        rng = np.random.default_rng(0)
        for i, row_key in enumerate(df.index):
            for j, col in enumerate(df.columns):
                arr = np.asarray(df.loc[row_key, col])
                offset = (j - (n_cols - 1) / 2) * strip_width
                x = i + offset + rng.uniform(-jitter * strip_width, jitter * strip_width, size=len(arr))
                color = cycle[j % len(cycle)]
                Axes.scatter(self, x, arr, s=s, alpha=alpha, color=color, **kwargs)

        # Set x labels to index names
        self.set_xticks(range(n_rows))
        self.set_xticklabels([str(k) for k in df.index])

        # Single column: ylabel instead of legend
        if n_cols == 1:
            self.set_ylabel(str(df.columns[0]))

        # Legend for column colors
        for j, col in enumerate(df.columns):
            color = cycle[j % len(cycle)]
            Axes.scatter(self, [], [], s=s, alpha=alpha, color=color, label=str(col))
        self._update_legend()

    def hist(self, data, bins=30, log_x=False, log_y=False, alpha=0.5, color=None,
             label=None, legend=True, **kwargs):
        """Overlaid histograms with shared bins.

        Pools all groups to compute a single bin edge array, then plots each
        group as a filled histogram on top of the others with alpha blending.
        This keeps bars directly comparable across groups.

        Parameters
        ----------
        data : DataFrame, dict, list of array-like, or 1-d array
            DataFrame: each column is one histogram.
            Dict: keys label each histogram.
            List/tuple of arrays: each element is one histogram.
            1-d array: a single histogram.
        bins : int or array-like, default 30
            Number of bins (int) or explicit bin edges.
        log_x : bool, default False
            If True, use log-spaced bins and set the x-axis to log scale.
            Requires all data to be positive. (Note: this differs from
            matplotlib's ``log=`` kwarg, which sets the *y*-axis to log.
            For log y, call ``ax.set_yscale('log')`` after ``hist``.)
        alpha : float, default 0.5
            Per-histogram alpha so overlapped bins blend visibly.
        color : str, list, or None, default None
            If None, use the color cycle (one color per group). A string
            paints every group the same; a list assigns colors positionally.
        label : str, list, or None, default None
            Legend labels. If None and data is a dict/DataFrame, the
            keys/columns are used automatically.
        **kwargs
            Additional keyword arguments passed to matplotlib hist()
            (e.g. ``density=True``, ``histtype='stepfilled'``).

        Returns
        -------
        list of (counts, edges, patches) tuples, one per group.
        """
        if not legend:
            self._cleanplots_no_legend = True
        cycle = get_color_cycle()

        # Normalize input to (groups, auto_labels)
        if _is_dataframe(data):
            groups = [np.asarray(data[col].dropna().values) for col in data.columns]
            auto_labels = [str(c) for c in data.columns]
        elif isinstance(data, dict):
            groups = [np.asarray(v) for v in data.values()]
            auto_labels = [str(k) for k in data.keys()]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                groups = [data]
            elif data.ndim == 2:
                groups = [data[i] for i in range(data.shape[0])]
            else:
                raise ValueError("data array must be 1-d or 2-d")
            auto_labels = None
        else:
            # list / tuple: either a flat sequence of numbers or a sequence of arrays
            first = data[0] if len(data) > 0 else None
            if np.isscalar(first):
                groups = [np.asarray(data)]
            else:
                groups = [np.asarray(g) for g in data]
            auto_labels = None

        n = len(groups)

        if color is None:
            group_colors = [cycle[i % len(cycle)] for i in range(n)]
        elif isinstance(color, (list, tuple, np.ndarray)):
            group_colors = [color[i % len(color)] for i in range(n)]
        else:
            group_colors = [color] * n

        if label is None:
            group_labels = list(auto_labels) if auto_labels is not None else [None] * n
        elif isinstance(label, (list, tuple)):
            group_labels = list(label)
        else:
            group_labels = [label] + [None] * (n - 1)

        # Shared bin edges from pooled data
        if isinstance(bins, (int, np.integer)):
            all_data = np.concatenate([g.ravel() for g in groups])
            lo = float(np.min(all_data))
            hi = float(np.max(all_data))
            if log_x:
                if lo <= 0:
                    raise ValueError("log_x=True requires all data to be positive")
                bin_edges = np.logspace(np.log10(lo), np.log10(hi), int(bins) + 1)
            else:
                bin_edges = np.linspace(lo, hi, int(bins) + 1)
        else:
            bin_edges = np.asarray(bins)

        # Use stepfilled so each histogram is a single filled polygon (not
        # per-bar rectangles). One hist() call with a list of arrays plus a
        # list of colors draws them overlaid at the same bin edges.
        kwargs.setdefault('histtype', 'stepfilled')

        result = Axes.hist(self, groups, bins=bin_edges, alpha=alpha,
                           color=group_colors, label=group_labels, **kwargs)

        if log_x:
            self.set_xscale('log')
        if log_y:
            self.set_yscale('log')
        return result

mprojections.register_projection(CleanAxes)


def get_color_cycle():
    """
    Returns the current color cycle as a list of hex codes.
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def _convertible_color(handle):
    """If this legend handle represents a single color, return it as an RGB tuple; else None.

    A handle is "convertible to a colored text label" if it's a solid Line2D, a
    uniformly-colored PathCollection (scatter), a uniformly-colored BarContainer,
    or a solid Patch, with a non-black color. Linestyle dummies (pivot column
    legends) and handles tagged with _cleanplots_skip_color_label (e.g. error
    band patches) return None.
    """
    if getattr(handle, '_cleanplots_skip_color_label', False):
        return None

    if isinstance(handle, Line2D):
        if handle.get_linestyle() != '-':
            return None
        rgb = mpl.colors.to_rgb(handle.get_color())
        if rgb == (0, 0, 0):
            return None
        return rgb

    if isinstance(handle, PathCollection):
        fc = handle.get_facecolor()
        if len(fc) == 0:
            return None
        first = tuple(fc[0][:3])
        if not all(tuple(c[:3]) == first for c in fc):
            return None
        if first == (0, 0, 0):
            return None
        return first

    if isinstance(handle, BarContainer):
        if not handle.patches:
            return None
        first = tuple(handle.patches[0].get_facecolor()[:3])
        for p in handle.patches:
            if tuple(p.get_facecolor()[:3]) != first:
                return None
        if first == (0, 0, 0):
            return None
        return first

    if isinstance(handle, Patch):
        fc = handle.get_facecolor()
        if fc is None or (len(fc) >= 4 and fc[3] == 0):
            return None
        rgb = tuple(fc[:3])
        if rgb == (0, 0, 0):
            return None
        return rgb

    return None


def _apply_auto_color_labels(ax):
    """Convert single-color legend entries into colored text inside a unified Legend.

    Reorders entries so all convertible (colored text) entries appear first,
    followed by non-convertible ones (linestyle dummies, error band patches,
    errorbar containers). For mixed legends, the handlebox width is zeroed out
    for convertible rows so their text is flush-left with the handle column of
    the other rows. All entries live in a single matplotlib Legend so the user
    can reposition it with the standard loc= API.
    """
    handles, labels = ax.get_legend_handles_labels()
    extra = getattr(ax, '_clean_extra_handles', [])
    all_handles = list(handles) + list(extra)
    all_labels = list(labels) + [h.get_label() for h in extra]
    if not all_labels:
        return

    entries = [(h, l, _convertible_color(h)) for h, l in zip(all_handles, all_labels)]
    convertible = [(l, c) for _, l, c in entries if c is not None]
    remaining = [(h, l) for h, l, c in entries if c is None]
    if len(convertible) < 2:
        return
    # Skip if all convertible colors are identical (color labels wouldn't disambiguate)
    if len({c for _, c in convertible}) < 2:
        return

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()

    # Reorder: convertible first, then the rest
    new_handles = [mpatches.Patch(facecolor='none', edgecolor='none') for _ in convertible]
    new_handles += [h for h, _ in remaining]
    new_labels = [l for l, _ in convertible] + [l for _, l in remaining]
    label_colors = [c for _, c in convertible] + ['black'] * len(remaining)

    all_convertible = not remaining
    legend_kw = dict(frameon=False, labelcolor=label_colors)
    if all_convertible:
        # Drop the handle column entirely
        legend_kw['handlelength'] = 0
        legend_kw['handletextpad'] = 0

    legend = Axes.legend(ax, new_handles, new_labels, **legend_kw)

    if not all_convertible:
        # Flush-left the colored text rows by zeroing their handlebox width
        # and their intra-row handle/text separator.
        n_convertible = len(convertible)
        for col in legend._legend_handle_box.get_children():
            for i, row in enumerate(col.get_children()):
                if i < n_convertible:
                    row_children = row.get_children()
                    if row_children:
                        row_children[0].width = 0
                    row.sep = 0


def _has_custom_labels(axis):
    """Check if an axis has a non-default formatter (e.g. string labels, custom formatting).

    Returns False for matplotlib's built-in defaults (ScalarFormatter for linear,
    LogFormatterSciNotation/LogFormatterMathtext for log scale) so clean() can
    override them. Returns True for user-set formatters (e.g. string labels).
    """
    default_types = (ScalarFormatter, LogFormatterSciNotation, LogFormatterMathtext)
    return not isinstance(axis.get_major_formatter(), default_types)

@overload
def fig(rows: Literal[1] = 1, cols: Literal[1] = 1, h: float | None = None, w: float | None = None, **kwargs) -> tuple[Figure, CleanAxes]: ...
@overload
def fig(rows: int, cols: int, h: float | None = None, w: float | None = None, **kwargs) -> tuple[Figure, np.ndarray]: ...

def fig(rows=1, cols=1, h=None, w=None, **kwargs):
    """Create a figure and axes with sensible default sizing.

    Parameters
    ----------
    rows : int, default 1
        Number of subplot rows.
    cols : int, default 1
        Number of subplot columns.
    h : float, optional
        Figure height in inches. Defaults to 4 * rows.
    w : float, optional
        Figure width in inches. Defaults to min(5 * cols, 10).
    **kwargs
        Additional keyword arguments passed to plt.subplots().

    Returns
    -------
    fig : Figure
    ax : CleanAxes or np.ndarray of CleanAxes
    """
    if w is None:
        w = min(5 * cols, 10)
    if h is None:
        h = 4 * rows
    subplot_kw = kwargs.pop('subplot_kw', {})
    subplot_kw.setdefault('projection', 'cleanplots')
    f, ax = plt.subplots(rows, cols, figsize=(w, h), subplot_kw=subplot_kw, **kwargs)
    return f, ax  # type: ignore[return-value]

##### General stylizing of plots #####
def clean(ax, spines='bottom_left', ticks='sparse', zero_origin=False, decimals='auto',
          color_labels='auto', **kwargs):
    """Format an axes for clean, publication-ready plots.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to format.
    spines : str or None, default 'bottom_left'
        Which spines to keep visible.
        - 'bottom_left' : only bottom and left spines
        - 'all' : all four spines
        - 'none' or None : no spines
    ticks : str, False, or None, default 'sparse'
        Tick density on axes.
        - 'sparse' : only min and max ticks on both axes
        - 'x' : sparse on x-axis only
        - 'y' : sparse on y-axis only
        - False : keep all default ticks
        - None : remove all ticks and tick labels (e.g. for images)
    zero_origin : bool or str, default False
        Whether axis limits start at zero.
        - True : both axes start at 0
        - 'x' : only x-axis starts at 0
        - 'y' : only y-axis starts at 0
        - False : don't modify axis limits
    decimals : str, int, or False, default 'auto'
        Tick label number formatting.
        - 'auto' : integers as "5", floats with 1 decimal as "3.5"
        - int : fixed decimal places (e.g. 2 gives "3.50"), integers still show without decimals
        - 'sci' : scientific notation (e.g. "1.2e+03")
        - False : keep matplotlib default formatting
    color_labels : 'auto' or False, default 'auto'
        If 'auto', convert single-color legend entries (solid lines, uniformly
        colored scatters) into inline color labels. Linestyle dummies and error
        band patches stay in a smaller legend below. Set to False to keep all
        entries in a normal legend.
    **kwargs
        Additional keyword arguments passed to ax.set() (e.g. xlabel, ylabel, title, xlim, ylim).
    """
    # Handle arrays/lists of axes (e.g. from plt.subplots(2, 3))
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            clean(a, spines=spines, ticks=ticks, zero_origin=zero_origin,
                  decimals=decimals, color_labels=color_labels, **kwargs)
        return
    if isinstance(ax, (list, tuple)):
        for a in ax:
            clean(a, spines=spines, ticks=ticks, zero_origin=zero_origin,
                  decimals=decimals, color_labels=color_labels, **kwargs)
        return

    x_custom = _has_custom_labels(ax.xaxis)
    y_custom = _has_custom_labels(ax.yaxis)
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if decimals is not False:
            _decimal_format_ticks(ax, decimals=decimals, skip_x=x_custom, skip_y=y_custom)
        if ticks:
            mode = 'both' if ticks == 'sparse' else ticks
            _sparse_ticks(ax, mode=mode, skip_x=x_custom, skip_y=y_custom)
    if spines == 'bottom_left':
        _clear_spines(ax)
    elif spines == 'none' or spines is None:
        _clear_spines(ax, all=True)
    # spines == 'all' → do nothing (keep all spines)
    if zero_origin is True:
        _zero_lims(ax)
    elif zero_origin in ('x', 'y'):
        _zero_lims(ax, mode=zero_origin)
    if getattr(ax, '_cleanplots_no_legend', False):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    else:
        if color_labels == 'auto':
            _apply_auto_color_labels(ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    if kwargs:
        ax.set(**kwargs)

def _default_format(ax, **kwargs):
    """Deprecated: use clean() instead."""
    clean(ax, zero_origin=True, **kwargs)

def _clear_spines(ax, all=False, leave=["bottom", "left"]):
    if type(ax) not in [list,  np.ndarray]:
        ax = [ax]
    for a in ax:
        if all:
            a.spines[["top", "right", "bottom", "left"]].set_visible(False)
        else:
            for spine in ["top", "right", "bottom", "left"]:
                if spine not in leave:
                    a.spines[spine].set_visible(False)
    
def _decimal_format_ticks(ax, decimals='auto', skip_x=False, skip_y=False):
    if decimals == 'auto':
        def formatter(x, pos):
            if x == int(x):
                return str(int(x))
            # Use g format for log-scale friendly display (e.g. 0.95 not 9.5e-01)
            formatted = f'{x:g}'
            if '.' not in formatted and 'e' not in formatted:
                return formatted
            return formatted
    elif decimals == 'sci':
        def formatter(x, pos):
            return f'{x:.1e}'
    elif isinstance(decimals, int):
        def formatter(x, pos):
            if x == int(x):
                return str(int(x))
            return f'{x:.{decimals}f}'
    if not skip_x:
        ax.xaxis.set_major_formatter(formatter)
    if not skip_y:
        ax.yaxis.set_major_formatter(formatter)
    
def _sparse_ticks(ax, mode='both', skip_x=False, skip_y=False):
    if mode in ('both', 'x') and not skip_x:
        xlim = ax.get_xlim()
        xticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
        if xticks:
            ax.set_xticks([xticks[0], xticks[-1]])
    if mode in ('both', 'y') and not skip_y:
        ylim = ax.get_ylim()
        yticks = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
        if yticks:
            ax.set_yticks([yticks[0], yticks[-1]])
        
def _zero_lims(ax, mode='both'):
    if mode == 'both':
        ax.set_xlim([0, ax.get_xlim()[-1]])
        ax.set_ylim([0, ax.get_ylim()[-1]])
    elif mode == 'x':
        ax.set_xlim([0, ax.get_xlim()[-1]])
    elif mode == 'y':
        ax.set_ylim([0, ax.get_ylim()[-1]])
        
    
