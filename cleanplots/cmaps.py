"""Curated colormaps organized by category.

Usage:
    import cleanplots as cp
    ax.heatmap(data, cmap=cp.cmaps.diverging)
    ax.heatmap(data, cmap=cp.cmaps.inferno)

Each category has a sensible default accessible as an attribute.
Individual named colormaps are also exposed directly.
"""

from matplotlib import colormaps as _colormaps
from matplotlib.colors import ListedColormap as _ListedColormap

# --- Sequential (low-to-high) ---
inferno = _colormaps['inferno']
plasma = _colormaps['plasma']
viridis = _colormaps['viridis']
magma = _colormaps['magma']
sequential = inferno  # default

# --- Diverging (centered, two-sided) ---
RdBu = _colormaps['RdBu']
coolwarm = _colormaps['coolwarm']
PiYG = _colormaps['PiYG']
diverging = RdBu  # default

# --- Binary/Mask ---
gray = _colormaps['gray']
binary = gray  # default

# --- Categorical (distinct colors, from our default cycle) ---
from cleanplots.cleanplots import colors as _colors
categorical = _ListedColormap(_colors, name='cleanplots_categorical')

# --- Cyclic/Phase ---
try:
    import cmocean as _cmocean
    phase = _cmocean.cm.phase
except ImportError:
    # Fallback if cmocean not installed
    phase = _colormaps['twilight']

# --- Crameri Scientific Colour Maps ---
try:
    from cmcrameri import cm as crameri
    from matplotlib.colors import Colormap as _Colormap
    for _name in dir(crameri):
        if _name.startswith('_'):
            continue
        _obj = getattr(crameri, _name)
        if isinstance(_obj, _Colormap):
            globals()[_name] = _obj
    del _name, _obj, _Colormap
except ImportError:
    crameri = None
