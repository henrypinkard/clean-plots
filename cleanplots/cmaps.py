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

# --- Isoluminant (constant lightness; order carried by hue alone) ---
# The dark-background choice: every level equally visible, no dark end to
# vanish. Default CET-I3 (cyan -> magenta) — no green to collide with the
# categorical cycle.
try:
    import colorcet as _colorcet
    CET_I1 = _colorcet.cm.CET_I1
    CET_I2 = _colorcet.cm.CET_I2
    CET_I3 = _colorcet.cm.CET_I3
except ImportError:
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    CET_I1 = CET_I2 = None
    CET_I3 = _LSC.from_list('CET_I3', [
        '#13b9e5', '#3fb6e7', '#58b4e9', '#6cb1eb', '#7daeec', '#8daaeb',
        '#9ca7ea', '#aba3e7', '#b89fe1', '#c59bdb', '#d098d4', '#da94cc',
        '#e490c4', '#ed8cbd', '#f588b5', '#fd84ac',
    ])
isolum = CET_I3  # default

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
