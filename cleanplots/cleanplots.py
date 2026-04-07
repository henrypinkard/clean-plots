#convenience code for plot export

import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.projections as mprojections
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.font_manager as fm
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation, LogFormatterMathtext

__all__ = [
    'fig',
    'clean',
    'colors',
    'get_color_cycle',
]

# Default color cycle
# https://davidmathlogic.com/colorblind/#%23179EE8-%235A00A0-%2321CA10-%23FF005B-%23D40E9F-%235A5A5A-%23DCCC02-%23FF7400
colors =  ['#179EE8', '#5A00A0', '#FF005B',  '#57B50F',
           '#D900FF', '#00E0E0', '#F37C2F', '#ACAD9D',]

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


class CleanAxes(Axes):
    """Axes subclass with convenience methods for clean plotting."""
    name = 'cleanplots'

    def clean(self, **kwargs):
        """Format this axes. See module-level clean() for full documentation."""
        clean(self, **kwargs)

    def _update_legend(self):
        """Rebuild legend if any labels exist."""
        handles, labels = self.get_legend_handles_labels()
        extra = getattr(self, '_clean_extra_handles', [])
        all_handles = handles + extra
        all_labels = labels + [h.get_label() for h in extra]
        if all_labels:
            super().legend(all_handles, all_labels)

    def line(self, x, y=None, err=None, label=None, err_label=None, alpha=0.2, **kwargs):
        """Plot a line with optional shaded confidence interval.

        Parameters
        ----------
        x : array-like, Series, or DataFrame
            If array-like, the x coordinates (y must also be provided).
            If Series, uses index as x and values as y.
            If DataFrame, plots each column as a separate line using the index
            as x and column names as labels. xlabel is inferred from index.name.
        y : array-like, optional
            The y coordinates. Required when x is array-like, ignored for
            Series/DataFrame.
        err : array-like, tuple of (low, high), Series, or DataFrame, optional
            Error data for shaded confidence intervals.
            For array mode: symmetric error or (low, high) bounds.
            For DataFrame mode: a DataFrame with matching columns, where each
            column provides the error for the corresponding column in x.
        label : str, optional
            Legend label for the line. Ignored in DataFrame mode (column names
            are used). For Series mode, defaults to series.name.
        err_label : str, optional
            Legend label for the error band (e.g. '±1 SE', '95% CI').
            In DataFrame mode, automatically shown only once.
        alpha : float, default 0.2
            Opacity of the shaded error region.
        **kwargs
            Additional keyword arguments passed to plot() (e.g. color, linewidth, marker).
        """
        # DataFrame mode: plot each column as a line
        if _is_dataframe(x):
            df = x
            all_lines = []
            for i, col in enumerate(df.columns):
                col_err = err[col] if (_is_dataframe(err) and col in err.columns) else None
                col_err_label = err_label if (i == 0 and err_label is not None) else None
                lines = self._line_array(df.index, df[col].values, err=col_err,
                                         label=str(col), err_label=col_err_label,
                                         alpha=alpha, **kwargs)
                all_lines.extend(lines)
            if hasattr(df.index, 'name') and df.index.name:
                self.set_xlabel(str(df.index.name))
            self._update_legend()
            return all_lines

        # Series mode: use index as x, values as y
        if _is_series(x):
            series = x
            lbl = label if label is not None else (str(series.name) if series.name is not None else None)
            err_vals = err.values if _is_series(err) else err
            if hasattr(series.index, 'name') and series.index.name:
                self.set_xlabel(str(series.index.name))
            return self._line_array(series.index, series.values, err=err_vals,
                                    label=lbl, err_label=err_label, alpha=alpha, **kwargs)

        # Array mode
        return self._line_array(x, y, err=err, label=label, err_label=err_label,
                                alpha=alpha, **kwargs)

    def _line_array(self, x, y, err=None, label=None, err_label=None, alpha=0.2, **kwargs):
        """Plot a single line with optional shaded confidence interval (array data)."""
        lines = self.plot(x, y, label=label, **kwargs)
        color = lines[0].get_color()
        if err is not None:
            if isinstance(err, tuple):
                low, high = np.asarray(err[0]), np.asarray(err[1])
            else:
                y = np.asarray(y)
                err = np.asarray(err)
                low, high = y - err, y + err
            self.fill_between(x, low, high, alpha=alpha, color=color,
                              label='_nolegend_')
            if err_label is not None:
                band_handle = mpatches.Patch(facecolor='gray', alpha=alpha, label=err_label)
                if not hasattr(self, '_clean_extra_handles'):
                    self._clean_extra_handles = []
                self._clean_extra_handles.append(band_handle)
        self._update_legend()
        return lines

    def bar(self, x, height=None, err=None, err_label=None, capsize=4, err_kw=None, **kwargs):
        """Plot a bar chart with optional error bars.

        Parameters
        ----------
        x : array-like, Series, or DataFrame
            If array-like, bar positions or labels (height must be provided).
            If Series, uses index as labels and values as heights.
            If DataFrame, plots grouped bars with one group per column, using
            the index as labels and column names as legend entries.
        height : array-like, optional
            Bar heights. Required when x is array-like, ignored for
            Series/DataFrame.
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
            lbl = kwargs.pop('label', None) or (str(series.name) if series.name is not None else None)
            err_vals = err.values if _is_series(err) else err
            if hasattr(series.index, 'name') and series.index.name:
                self.set_xlabel(str(series.index.name))
            return self._bar_array([str(l) for l in series.index], series.values,
                                   err=err_vals, err_label=err_label, capsize=capsize,
                                   err_kw=err_kw, label=lbl, **kwargs)

        # Array mode
        return self._bar_array(x, height, err=err, err_label=err_label,
                               capsize=capsize, err_kw=err_kw, **kwargs)

    def _bar_array(self, x, height, err=None, err_label=None, capsize=4, err_kw=None, **kwargs):
        """Plot bars from array data."""
        if err_kw is None:
            err_kw = {}
        err_kw.setdefault('color', 'black')
        err_kw.setdefault('linewidth', 1.5)
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

    def scatter(self, x, y, xerr=None, yerr=None, err_label=None, capsize=4, err_kw=None,
                 label=None, **kwargs):
        """Scatter plot with optional error bars in x and/or y.

        Parameters
        ----------
        x, y : array-like
            Data coordinates.
        xerr : array-like or tuple of (low, high), optional
            Horizontal error bars. Symmetric if array-like, asymmetric if tuple.
        yerr : array-like or tuple of (low, high), optional
            Vertical error bars. Symmetric if array-like, asymmetric if tuple.
        err_label : str, optional
            Legend label for the error bars (e.g. '±1 SE').
            Only set this on one call to avoid duplicate legend entries.
        capsize : float, default 4
            Width of error bar caps in points.
        err_kw : dict, optional
            Additional keyword arguments for the error bars.
        label : str, optional
            Legend label for the scatter points.
        **kwargs
            Additional keyword arguments passed to matplotlib scatter()
            (e.g. color, s, marker).
        """
        sc = super().scatter(x, y, label=label, **kwargs)
        if xerr is not None or yerr is not None:
            if err_kw is None:
                err_kw = {}
            err_kw.setdefault('fmt', 'none')
            err_kw.setdefault('color', 'black')
            err_kw.setdefault('linewidth', 1.5)
            err_kw.setdefault('capsize', capsize)
            if isinstance(xerr, tuple):
                xerr = np.array([np.asarray(xerr[0]), np.asarray(xerr[1])])
            if isinstance(yerr, tuple):
                yerr = np.array([np.asarray(yerr[0]), np.asarray(yerr[1])])
            self.errorbar(x, y, xerr=xerr, yerr=yerr, **err_kw)
            if err_label is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.errorbar([np.nan], [np.nan], yerr=[0.5], fmt='none',
                                  color=err_kw['color'], capsize=capsize,
                                  linewidth=err_kw['linewidth'], label=err_label)
        self._update_legend()
        return sc

    def legend(self, *args, **kwargs):
        """Override legend to include extra handles from line()/bar()/scatter() err_label."""
        extra = getattr(self, '_clean_extra_handles', [])
        if extra:
            existing_handles, existing_labels = self.get_legend_handles_labels()
            handles = existing_handles + extra
            labels = existing_labels + [h.get_label() for h in extra]
            return super().legend(handles, labels, *args, **kwargs)
        return super().legend(*args, **kwargs)

mprojections.register_projection(CleanAxes)


def get_color_cycle():
    """
    Returns the current color cycle as a list of hex codes.
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def _has_custom_labels(axis):
    """Check if an axis has a non-default formatter (e.g. string labels, custom formatting).

    Returns False for matplotlib's built-in defaults (ScalarFormatter for linear,
    LogFormatterSciNotation/LogFormatterMathtext for log scale) so clean() can
    override them. Returns True for user-set formatters (e.g. string labels).
    """
    default_types = (ScalarFormatter, LogFormatterSciNotation, LogFormatterMathtext)
    return not isinstance(axis.get_major_formatter(), default_types)

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
    fig, ax : matplotlib Figure and Axes
    """
    if w is None:
        w = min(5 * cols, 10)
    if h is None:
        h = 4 * rows
    subplot_kw = kwargs.pop('subplot_kw', {})
    subplot_kw.setdefault('projection', 'cleanplots')
    return plt.subplots(rows, cols, figsize=(w, h), subplot_kw=subplot_kw, **kwargs)

##### General stylizing of plots #####
def clean(ax, spines='bottom_left', ticks='sparse', zero_origin=False, decimals='auto', **kwargs):
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
    **kwargs
        Additional keyword arguments passed to ax.set() (e.g. xlabel, ylabel, title, xlim, ylim).
    """
    # Handle arrays/lists of axes (e.g. from plt.subplots(2, 3))
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            clean(a, spines=spines, ticks=ticks, zero_origin=zero_origin, decimals=decimals, **kwargs)
        return
    if isinstance(ax, (list, tuple)):
        for a in ax:
            clean(a, spines=spines, ticks=ticks, zero_origin=zero_origin, decimals=decimals, **kwargs)
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
        
    
