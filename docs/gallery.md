A reference for the cleanplots-specific API. Assumes matplotlib familiarity.

```python
import numpy as np
import pandas as pd
import cleanplots as cp
import matplotlib.pyplot as plt
```

---

## Setup and common patterns

Every plot starts with `cp.fig()` and ends with `ax.clean()`.

```python
f, ax = cp.fig()                          # single axes
f, axes = cp.fig(rows=2, cols=2)          # grid
f, (ax1, ax2) = cp.fig(cols=2)            # side-by-side
ax.clean(xlabel='X', ylabel='Y')          # standard cleanup
ax.clean(ticks=None, spines=None)         # no axes (for images)
```

**Error representation.** `err=` adds error visualization (shaded band for lines, error bars for bar/scatter). Accepts symmetric error, a `(lower, upper)` tuple, or an Nx2 array. `err_label=` creates a single shared legend entry.

**Legend.** `clean()` builds the legend automatically. Pass `legend=False` to any plot call to suppress it entirely. To reposition:

```python
ax.get_legend().set_loc('upper left')     # reposition after clean()
```

**Color labels.** `clean()` automatically replaces multi-color legend entries with colored text. To place a color-label block manually:

```python
ax.color_labels(['MLP', 'RF', 'Transformer'], x=0.05, ha='left')
```

<img src="figures/color_labels.png" width="30%">

---

## Line plots

### Training curves with error bands

Starting from tidy per-epoch data:

```
     model    lr  seed  epoch  train_loss  val_loss
0    resnet  0.01     0      0        0.82      0.91
1    resnet  0.01     0      1        0.54      0.63
...
```

Aggregate over seeds with `cp.data.collapse`:

```python
mean = cp.data.collapse(df, 'epoch', ['train_loss', 'val_loss'], over='seed', func='mean')
std  = cp.data.collapse(df, 'epoch', ['train_loss', 'val_loss'], over='seed', func='std')
```

The resulting DataFrame has MultiIndex columns. The top level (`metric`) maps to line styles, remaining levels (`model`, `lr`) map to colors:

```
metric     train_loss                    val_loss
model      resnet       mlp              resnet       mlp
lr         0.01  0.001  0.01  0.001      0.01  0.001  0.01  0.001
epoch
0          0.82  0.91   1.01  1.12       0.91  1.02   1.12  1.24
1          0.54  0.63   0.72  0.81       0.63  0.73   0.82  0.92
...
```

Use `cp.data.slice` to filter, then plot:

```python
f, (ax1, ax2) = cp.fig(cols=2, sharey=True)
ax1.line(cp.data.slice(mean, lr=0.001), err=cp.data.slice(std, lr=0.001), err_label='±1 SD')
ax1.clean(ylabel='Loss', title='lr 0.001')
ax2.line(cp.data.slice(mean, lr=0.01), err=cp.data.slice(std, lr=0.01), legend=False)
ax2.clean(title='lr 0.01')
```

<img src="figures/wide_multi_line.png" width="30%">

### Plotting individual runs

Pass a single `color=` and `label=` to overlay multiple same-colored lines. When `color=` and `label=` are both set on a DataFrame, you get one color legend entry plus shared linestyle entries:

```python
wide = cp.data.to_wide(df, 'epoch', ['train_loss', 'val_loss'])

for i, model in enumerate(['small', 'medium', 'large']):
    ax.line(cp.data.slice(wide, model=model), color=cp.colors[i], label=model, alpha=0.5)
ax.clean()
```

For raw arrays (e.g. RL reward curves), a list of arrays works directly:

```python
ax.line(runs, color=cp.colors[0], alpha=0.3, label='Runs')
ax.line(np.mean(runs, axis=0), color=cp.colors[0], lw=2, label='Mean')
ax.clean(xlabel='Step', ylabel='Reward')
```

<img src="figures/multi_run.png" width="30%">

---

## Bar charts

### Grouped bar chart with error bars

Starting from tidy per-seed metrics:

```
     model           seed  precision  recall    f1
0    random_forest   0     0.92       0.88      0.90
1    random_forest   1     0.90       0.86      0.88
...
```

Collapse over seeds:

```python
mean = cp.data.collapse(df, 'model', ['precision', 'recall', 'f1'], over='seed', func='mean')
std  = cp.data.collapse(df, 'model', ['precision', 'recall', 'f1'], over='seed', func='std')
```

Result — index becomes x-axis categories, columns become colored groups:

```
metric          precision  recall    f1
model
gradient_boost  0.88       0.92      0.90
logistic        0.82       0.85      0.83
mlp             0.94       0.90      0.92
random_forest   0.91       0.87      0.89
```

```python
f, ax = cp.fig()
ax.bar(mean, err=std, err_label='±1 SD')
ax.clean(ylabel='Score')
```

<img src="figures/grouped_bar.png" width="30%">

### Horizontal bar chart

For single-metric results, `collapse` returns a Series. Pass it directly to `barh`:

```python
mean = cp.data.collapse(df, 'feature', 'importance', over='seed', func='mean')
std  = cp.data.collapse(df, 'feature', 'importance', over='seed', func='std')

f, ax = cp.fig()
ax.barh(mean, err=std)
ax.clean()
```

<img src="figures/bar_chart.png" width="30%">

---

## Scatter

`yerr=` and `xerr=` add error bars. Use `err_label=`, `xerr_label=`, `yerr_label=` for legend entries.

```python
f, ax = cp.fig()
ax.scatter(x, y, xerr=xerrs, yerr=yerrs, label='Model A',
           xerr_label='time uncertainty', yerr_label='accuracy uncertainty')
ax.clean(xlabel='Training time (min)', ylabel='Accuracy')
```

<img src="figures/scatter_xy_errorbars.png" width="30%">

An Nx2 array or DataFrame splits into x, y (DataFrame column names become axis labels); Nx3 uses the third column as marker size.

```python
ax.scatter(df[['PC1', 'PC2']], s=15, alpha=0.6)
```

<img src="figures/scatter_clusters.png" width="30%">

### Categorical grouping

`group=` assigns colors from the cycle and creates legend entries. Mutually exclusive with `c=`. Array-valued kwargs like `edgecolors` are automatically split per group.

```python
f, ax = cp.fig()
ax.scatter(df['x'], df['y'], group=df['class'],
           edgecolors=cp.cmaps.inferno(df['confidence']), linewidths=1.5)
cb = ax.add_colorbar(cp.cmaps.inferno, values=df['confidence'])
cb.set_label('confidence')
ax.clean(xlabel='x', ylabel='y')
```

### Colorbar for manual color mapping

`ax.add_colorbar(cmap, values)` creates a colorbar from a colormap and the data values that were mapped through it. Works on any axes, not just scatter.

```python
cb = ax.add_colorbar(cp.cmaps.inferno, values=data)   # range from data
cb = ax.add_colorbar(cp.cmaps.inferno, vmin=0, vmax=1) # explicit range
cb.set_label('confidence')
cb.set_ticks([0, 0.5, 1])
```

---

## Heatmap

### Hyperparameter grid search

Starting from tidy cross-validation results:

```
     C      gamma  fold  test_accuracy
0    0.01   0.001  0     0.71
1    0.01   0.001  1     0.73
...
```

Collapse over folds — the resulting DataFrame's index and columns become axis labels:

```python
mean = cp.data.collapse(df, 'gamma', 'test_accuracy', over='fold', func='mean')
std  = cp.data.collapse(df, 'gamma', 'test_accuracy', over='fold', func='std')
```

```
mean test_accuracy
C            0.01   0.1    1.0    10.0   100.0
gamma
0.001        0.72   0.74   0.78   0.82   0.79
0.01         0.73   0.76   0.83   0.90   0.85
0.1          0.71   0.73   0.77   0.81   0.78
1.0          0.70   0.71   0.72   0.73   0.72
```

```python
annot = mean.round(2).astype(str) + '±' + std.round(2).astype(str)

f, ax = cp.fig()
ax.heatmap(mean, annot_strings=annot, cmap='Blues')
ax.clean(ticks=False, spines=None)
```

<img src="figures/confusion_matrix.png" width="30%">

`heatmap` returns the matplotlib `Colorbar` object (or `None` if `cbar=False`), so you can set its label, ticks, etc. directly:

```python
cb = ax.heatmap(data, cmap='Blues')
cb.set_label('Count')
cb.set_ticks([0, 25, 50])
```

---

## Comparing distributions

`box`, `violin`, `strip`, and `hist` all take a dict of group name → array (or DataFrame / list of arrays).

### Grouped layout from a DataFrame

To show distributions side-by-side (like a grouped bar chart but with raw data points), use `collapse` with `func=list` to collect raw values into arrays, then pass the resulting DataFrame:

```python
df = cp.data.collapse(data, 'model', ['precision', 'recall', 'f1'], over='seed', func=list)
```

```
metric          precision               recall                  f1
model
gradient_boost  [0.88, 0.87, 0.89, ...]  [0.92, 0.91, 0.93, ...]  [0.90, 0.89, ...]
logistic        [0.82, 0.81, 0.83, ...]  [0.85, 0.84, 0.86, ...]  [0.83, 0.82, ...]
mlp             [0.94, 0.93, 0.95, ...]  [0.90, 0.89, 0.91, ...]  [0.92, 0.91, ...]
random_forest   [0.91, 0.90, 0.92, ...]  [0.87, 0.86, 0.88, ...]  [0.89, 0.88, ...]
```

Index → x-axis categories, columns → colored groups (matching `bar`):

```python
ax.strip(df)     # or ax.box(df), or ax.violin(df)
```

<img src="figures/box_plot.png" width="22%"> <img src="figures/violin_plot.png" width="22%">

<img src="figures/strip_plot.png" width="30%">

### Histograms

`hist` draws overlaid histograms with shared bins. Uses `histtype='stepfilled'` and `alpha=0.5` by default.

```python
f, ax = cp.fig()
ax.hist([baseline, tuned, ablation],
        label=['Baseline', 'Tuned', 'Ablation'],
        bins=40, density=True)
ax.clean(xlabel='Prediction error', ylabel='Density')
```

<img src="figures/hist_overlay.png" width="30%">

`log_x=True` gives log-spaced bins on a log x-axis.

<img src="figures/hist_overlay_log.png" width="30%">

---

## Multi-panel layouts

```python
f, axes = cp.fig(rows=2, cols=2)
axes[0, 0].line(loss);                    axes[0, 0].clean(ylabel='Loss', xlabel='Epoch')
axes[0, 1].bar(['A', 'B', 'C'], vals);    axes[0, 1].clean(ylabel='Accuracy')
axes[1, 0].scatter(embeddings, s=20);     axes[1, 0].clean(xlabel='PC1', ylabel='PC2')
axes[1, 1].box({'M1': s1, 'M2': s2});     axes[1, 1].clean(ylabel='Score')
f.tight_layout()
```

<img src="figures/grid_layout.png" width="30%">

```python
f.subplots_adjust(hspace=0.55, wspace=0.45)   # when labels overlap
```

<img src="figures/grid_layout_spaced.png" width="30%">

---

## Data wrangling (`cp.data`)

Helpers for the tidy → wide → aggregated pipeline common in ML experiments. Loud errors over silent wrong numbers (uses `pivot`, not `pivot_table`).

```python
# One-step: pivot + aggregate
mean = cp.data.collapse(df, index='epoch', values=['train_loss', 'val_loss'],
                         over='seed', func='mean')

# Two-step equivalent:
wide = cp.data.to_wide(df, index='epoch', values=['train_loss', 'val_loss'])
mean = cp.data.agg_over(wide, col_level='seed', func='mean')

# Filter by column level
cp.data.slice(mean, model='resnet')
cp.data.slice(mean, metric='train_loss', lr=0.01)
```

---

## Colormaps

Access via `cp.cmaps.<name>`.

| Category | Default | Other options |
|---|---|---|
| Sequential | <img src="figures/cmap_inferno.png" width="60"> `inferno` | <img src="figures/cmap_plasma.png" width="40"> `plasma` &nbsp; <img src="figures/cmap_viridis.png" width="40"> `viridis` &nbsp; <img src="figures/cmap_magma.png" width="40"> `magma` |
| Diverging | <img src="figures/cmap_RdBu.png" width="60"> `RdBu` | <img src="figures/cmap_coolwarm.png" width="40"> `coolwarm` &nbsp; <img src="figures/cmap_PiYG.png" width="40"> `PiYG` |
| Binary | <img src="figures/cmap_gray.png" width="60"> `gray` | — |
| Categorical | <img src="figures/cmap_categorical.png" width="60"> `categorical` | (built from `cp.colors`) |
| Cyclic | <img src="figures/cmap_phase.png" width="60"> `phase` (cmocean) | — |

---

## Hard-to-Google tricks

Matplotlib features that cleanplots doesn't wrap. Work on any `CleanAxes` (it's an `Axes` subclass).

### Rotate tick labels

```python
ax.tick_params(axis='x', rotation=45)
plt.setp(ax.get_xticklabels(), ha='right')
```

<img src="figures/trick_rotation.png" width="45%">

### Annotate a point

```python
ax.annotate('best',
            xy=(150, 0.93), xytext=(100, 0.80),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
```

<img src="figures/trick_annotate.png" width="30%">

### Square axes

```python
ax.set_aspect('equal')
```

<img src="figures/trick_aspect_equal.png" width="45%">
