"""Generate pure-matplotlib versions of the gallery figures.

Companion to generate_gallery.py. Each example here reproduces (as closely as
pure matplotlib reasonably allows) a plot from the cleanplots gallery, so you
can fall back to raw matplotlib if cleanplots isn't available.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / 'matplotlib_figures'
FIGDIR.mkdir(exist_ok=True)

np.random.seed(42)


def clean_spines(ax):
    """Hide the top and right spines on an Axes. (Mirror of cleanplots' default.)"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# 1. LINE PLOT WITH ERROR BAND
# =============================================================================

x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
e1 = 0.2 + 0.1 * np.abs(np.sin(x))
e2 = 0.15 + 0.1 * np.abs(np.cos(x))

f, ax = plt.subplots(figsize=(5, 4))
line1, = ax.plot(x, y1, label='Model A')
ax.fill_between(x, y1 - e1, y1 + e1, alpha=0.2, color=line1.get_color(), label='±1 SD')
line2, = ax.plot(x, y2, label='Model B')
ax.fill_between(x, y2 - e2, y2 + e2, alpha=0.2, color=line2.get_color())
ax.set_xlabel('Step')
ax.set_ylabel('Score')
ax.legend(frameon=False)
clean_spines(ax)
f.savefig(FIGDIR / 'error_band.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 2. PIVOT DATAFRAME LINE PLOT (multi-model, multi-metric with error bands)
# =============================================================================

num_epochs = 30
models = ['MLP', 'Transformer']
train_histories = {
    'MLP': np.random.randn(5, num_epochs).cumsum(axis=1) * 0.1 + np.linspace(2.5, 0.5, num_epochs),
    'Transformer': np.random.randn(5, num_epochs).cumsum(axis=1) * 0.1 + np.linspace(2.0, 0.3, num_epochs),
}
val_histories = {
    'MLP': np.random.randn(5, num_epochs).cumsum(axis=1) * 0.1 + np.linspace(2.6, 0.7, num_epochs),
    'Transformer': np.random.randn(5, num_epochs).cumsum(axis=1) * 0.1 + np.linspace(2.1, 0.5, num_epochs),
}

pivot = pd.DataFrame({
    'train_loss': [train_histories[m].mean(axis=0) for m in models],
    'val_loss': [val_histories[m].mean(axis=0) for m in models],
}, index=models)
pivot_std = pd.DataFrame({
    'train_loss': [train_histories[m].std(axis=0) for m in models],
    'val_loss': [val_histories[m].std(axis=0) for m in models],
}, index=models)

f, ax = plt.subplots(figsize=(5, 4))
row_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles = ['-', '--', ':', '-.']
epochs_x = np.arange(num_epochs)

for i, model in enumerate(pivot.index):
    color = row_colors[i % len(row_colors)]
    for j, metric in enumerate(pivot.columns):
        ls = line_styles[j % len(line_styles)]
        y = np.asarray(pivot.loc[model, metric])
        e = np.asarray(pivot_std.loc[model, metric])
        ax.plot(epochs_x, y, color=color, linestyle=ls)
        ax.fill_between(epochs_x, y - e, y + e, color=color, alpha=0.2)

# Hand-built two-axis legend: one entry per model (color) and per metric (linestyle)
model_handles = [plt.Line2D([0], [0], color=row_colors[i], linestyle='-', label=m)
                 for i, m in enumerate(pivot.index)]
metric_handles = [plt.Line2D([0], [0], color='black', linestyle=line_styles[j], label=m)
                  for j, m in enumerate(pivot.columns)]
band_handle = plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.2, label='±1 SD')
ax.legend(handles=model_handles + metric_handles + [band_handle], frameon=False)
ax.set_xlabel('Epoch')
clean_spines(ax)
f.savefig(FIGDIR / 'pivot_line.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 3. GROUPED BAR CHART FROM DATAFRAME
# =============================================================================

bar_df = pd.DataFrame({
    'Precision': [0.91, 0.88, 0.95],
    'Recall':    [0.87, 0.92, 0.90],
    'F1':        [0.89, 0.90, 0.92],
}, index=pd.Index(['RF', 'GB', 'MLP'], name='Model'))
bar_err = pd.DataFrame({
    'Precision': [0.02, 0.03, 0.01],
    'Recall':    [0.03, 0.02, 0.02],
    'F1':        [0.02, 0.02, 0.01],
}, index=bar_df.index)

f, ax = plt.subplots(figsize=(5, 4))
n_groups = len(bar_df.columns)
x_pos = np.arange(len(bar_df.index))
width = 0.8 / n_groups
for i, col in enumerate(bar_df.columns):
    offset = (i - (n_groups - 1) / 2) * width
    ax.bar(x_pos + offset, bar_df[col].values, width=width,
           yerr=bar_err[col].values, capsize=4,
           error_kw=dict(linewidth=1.5), label=col)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_df.index)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.legend(frameon=False)
clean_spines(ax)
f.savefig(FIGDIR / 'grouped_bar.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 4. SCATTER WITH Y ERROR BARS
# =============================================================================

x_data = np.array([50, 100, 200, 500, 1000])
y_means = np.array([0.72, 0.81, 0.88, 0.92, 0.95])
y_errs = np.array([0.05, 0.04, 0.03, 0.02, 0.01])

f, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(x_data, y_means, yerr=y_errs, fmt='o', capsize=4,
            label='Model A')
ax.set_xlabel('Training samples')
ax.set_ylabel('Accuracy')
ax.legend(frameon=False)
clean_spines(ax)
f.savefig(FIGDIR / 'scatter_errorbars.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 5. SCATTER WITH BOTH X AND Y ERROR BARS
# =============================================================================

xy_x = np.array([10, 30, 60, 100, 150])
xy_y = 0.80 + 0.0010 * xy_x + np.random.normal(0, 0.008, len(xy_x))
xy_xerr = np.array([2, 5, 8, 10, 15])
xy_yerr = np.array([0.02, 0.018, 0.014, 0.012, 0.010])

f, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(xy_x, xy_y, xerr=xy_xerr, yerr=xy_yerr, fmt='o', capsize=4,
            label='Model A')
ax.set_xlabel('Training time (min)')
ax.set_ylabel('Accuracy')
ax.legend(frameon=False)
clean_spines(ax)
f.savefig(FIGDIR / 'scatter_xy_errorbars.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 6. HEATMAP (confusion matrix)
# =============================================================================

conf_matrix = pd.DataFrame(
    [[45, 3, 2], [4, 42, 4], [1, 5, 44]],
    index=['Cat', 'Dog', 'Bird'],
    columns=['Cat', 'Dog', 'Bird'],
)

f, ax = plt.subplots(figsize=(5, 4))
values = conf_matrix.values
im = ax.imshow(values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(conf_matrix.columns)))
ax.set_xticklabels(conf_matrix.columns)
ax.set_yticks(range(len(conf_matrix.index)))
ax.set_yticklabels(conf_matrix.index)
ax.xaxis.set_label_position('top')
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
# Annotate cells; flip text color for contrast
threshold = (values.min() + values.max()) / 2
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        color = 'white' if values[i, j] > threshold else 'black'
        ax.text(j, i, f'{values[i, j]:d}', ha='center', va='center', color=color)
f.colorbar(im, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
f.savefig(FIGDIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 7. BOX PLOT
# =============================================================================

box_data = {
    'LR':  np.random.normal(0.82, 0.03, 30),
    'RF':  np.random.normal(0.91, 0.02, 30),
    'GB':  np.random.normal(0.93, 0.015, 30),
    'MLP': np.random.normal(0.89, 0.04, 30),
}

f, ax = plt.subplots(figsize=(5, 4))
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
bp = ax.boxplot(list(box_data.values()), tick_labels=list(box_data.keys()),
                patch_artist=True)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cycle[i % len(cycle)])
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy')
clean_spines(ax)
f.savefig(FIGDIR / 'box_plot.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 7b. VIOLIN PLOT
# =============================================================================

f, ax = plt.subplots(figsize=(5, 4))
vp = ax.violinplot(list(box_data.values()), showmeans=False, showmedians=True)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(cycle[i % len(cycle)])
    body.set_alpha(0.7)
ax.set_xticks(range(1, len(box_data) + 1))
ax.set_xticklabels(list(box_data.keys()))
ax.set_ylabel('Accuracy')
clean_spines(ax)
f.savefig(FIGDIR / 'violin_plot.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 8. MULTI-PANEL LAYOUT
# =============================================================================

epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.3 + np.random.normal(0, 0.02, len(epochs))

f, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(epochs, train_loss)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')

axes[0, 1].bar(['A', 'B', 'C'], [0.9, 0.85, 0.95])
axes[0, 1].set_ylabel('Accuracy')

pts = np.random.randn(50, 2)
axes[1, 0].scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.6)
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

axes[1, 1].boxplot([np.random.randn(30), np.random.randn(30) + 0.5],
                   tick_labels=['M1', 'M2'])
axes[1, 1].set_ylabel('Score')

for a in axes.flat:
    clean_spines(a)
f.tight_layout()
f.savefig(FIGDIR / 'grid_layout.png', dpi=150, bbox_inches='tight')
plt.close(f)


print(f"Generated {len(list(FIGDIR.glob('*.png')))} figures in {FIGDIR}")
