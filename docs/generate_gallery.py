"""Generate all example figures for the cleanplots gallery."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import cleanplots as cp
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / 'figures'
FIGDIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# 1. LINE PLOTS
# =============================================================================

# --- 1a. Basic learning curves (train/val) ---
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.3 + np.random.normal(0, 0.02, len(epochs))
val_loss = 2.5 * np.exp(-0.06 * epochs) + 0.5 + np.random.normal(0, 0.03, len(epochs))

f, ax = cp.fig()
ax.line(epochs, train_loss, label='Train')
ax.line(epochs, val_loss, label='Val')
ax.clean(xlabel='Epoch', ylabel='Loss')
f.savefig(FIGDIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1b. Line with shaded error band ---
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
e1 = 0.2 + 0.1 * np.abs(np.sin(x))
e2 = 0.15 + 0.1 * np.abs(np.cos(x))

f, ax = cp.fig()
ax.line(x, y1, err=e1, label='Model A', err_label='±1 SD')
ax.line(x, y2, err=e2, label='Model B')
ax.clean(xlabel='Step', ylabel='Score')
f.savefig(FIGDIR / 'error_band.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1c. Multi-line from list ---
np.random.seed(42)
curves = [np.cumsum(np.random.randn(100)) for _ in range(3)]

f, ax = cp.fig()
ax.line(curves, label=['Run 1', 'Run 2', 'Run 3'])
ax.clean(xlabel='Step', ylabel='Cumulative reward')
f.savefig(FIGDIR / 'multi_line.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1c2. Many runs of one experiment, same color ---
np.random.seed(0)
runs = [np.cumsum(np.random.randn(50)) for _ in range(8)]

f, ax = cp.fig()
ax.line(runs, color=cp.colors[0], alpha=0.3, label='Runs')
ax.line(np.mean(runs, axis=0), color=cp.colors[0], lw=2, label='Mean')
ax.clean(xlabel='Step', ylabel='Reward')
f.savefig(FIGDIR / 'multi_run.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1d. Single-arg y-only ---
f, ax = cp.fig()
ax.line(train_loss, label='Train loss')
ax.clean(xlabel='Epoch', ylabel='Loss')
f.savefig(FIGDIR / 'single_arg_line.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1e. Pivot DataFrame (multi-model, multi-metric) ---
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

f, ax = cp.fig()
ax.line(pivot, err=pivot_std, err_label='±1 SD')
ax.clean(xlabel='Epoch')
f.savefig(FIGDIR / 'pivot_line.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1e2. Wide MultiIndex line from tidy data via cp.data ---
tidy_rows = []
for model in ['MLP', 'Transformer']:
    for seed in range(5):
        for epoch in range(num_epochs):
            tidy_rows.append({
                'model': model, 'seed': seed, 'epoch': epoch,
                'train_loss': train_histories[model][seed, epoch],
                'val_loss': val_histories[model][seed, epoch],
            })
tidy_df = pd.DataFrame(tidy_rows)
wide_mean = cp.data.collapse(tidy_df, 'epoch', ['train_loss', 'val_loss'], over='seed', func='mean')
wide_std = cp.data.collapse(tidy_df, 'epoch', ['train_loss', 'val_loss'], over='seed', func='std')

f, ax = cp.fig()
ax.line(wide_mean, err=wide_std, err_label='±1 SD')
ax.clean(ylabel='Loss')
f.savefig(FIGDIR / 'wide_multi_line.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1f. Pivot single column (ylabel inferred) ---
means_single = pd.DataFrame({
    'val_loss': [val_histories[m].mean(axis=0) for m in models],
}, index=models)
std_single = pd.DataFrame({
    'val_loss': [val_histories[m].std(axis=0) for m in models],
}, index=models)

f, ax = cp.fig()
ax.line(means_single, err=std_single, err_label='±1 SD')
ax.clean(xlabel='Epoch')
f.savefig(FIGDIR / 'pivot_single_col.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1g. DataFrame scalar columns ---
df_scalar = pd.DataFrame({
    'MLP': np.random.randn(20).cumsum(),
    'RF': np.random.randn(20).cumsum(),
    'GB': np.random.randn(20).cumsum(),
}, index=pd.RangeIndex(1, 21, name='Epoch'))

f, ax = cp.fig()
ax.line(df_scalar)
ax.clean(ylabel='Score')
f.savefig(FIGDIR / 'dataframe_line.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 1h. Side-by-side subplots (loss + accuracy) ---
train_acc = 1 - 0.5 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.01, len(epochs))
val_acc = 1 - 0.6 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.015, len(epochs))

f, (ax1, ax2) = cp.fig(cols=2)
ax1.line(epochs, train_loss, label='Train')
ax1.line(epochs, val_loss, label='Val')
ax1.clean(xlabel='Epoch', ylabel='Loss')

ax2.line(epochs, train_acc, label='Train')
ax2.line(epochs, val_acc, label='Val')
ax2.clean(xlabel='Epoch', ylabel='Accuracy')
f.savefig(FIGDIR / 'side_by_side.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 2. BAR CHARTS
# =============================================================================

# --- 2a. Simple bar chart with error ---
model_names = ['LR', 'RF', 'GB', 'MLP', 'Transformer']
accuracies = [0.82, 0.91, 0.93, 0.89, 0.95]
stds = [0.03, 0.02, 0.015, 0.04, 0.01]

f, ax = cp.fig()
ax.bar(model_names, accuracies, err=stds, err_label='±1 SD')
ax.clean(ylabel='Accuracy')
f.savefig(FIGDIR / 'bar_chart.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 2b. Grouped bar from DataFrame ---
bar_df = pd.DataFrame({
    'Precision': [0.91, 0.88, 0.95],
    'Recall': [0.87, 0.92, 0.90],
    'F1': [0.89, 0.90, 0.92],
}, index=pd.Index(['RF', 'GB', 'MLP'], name='Model'))
bar_err = pd.DataFrame({
    'Precision': [0.02, 0.03, 0.01],
    'Recall': [0.03, 0.02, 0.02],
    'F1': [0.02, 0.02, 0.01],
}, index=bar_df.index)

f, ax = cp.fig()
ax.bar(bar_df, err=bar_err, err_label='±1 SD')
ax.clean(ylabel='Score')
ax.get_legend().set_loc('upper left')
f.savefig(FIGDIR / 'grouped_bar.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 2c. Single-arg bar ---
f, ax = cp.fig()
ax.bar([0.82, 0.91, 0.93, 0.89, 0.95])
ax.clean(ylabel='Accuracy')
f.savefig(FIGDIR / 'single_arg_bar.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 2d. Horizontal bar (feature importances) ---
features = ['Feature D', 'Feature A', 'Feature C', 'Feature B', 'Feature E']
importances = sorted([0.35, 0.25, 0.20, 0.12, 0.08])

f, ax = cp.fig()
ax.barh(features, importances, color=cp.colors[0])
ax.clean(xlabel='Importance', spines='bottom_left')
f.savefig(FIGDIR / 'horizontal_bar.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 3. SCATTER PLOTS
# =============================================================================

# --- 3a. Basic scatter with error bars ---
x_data = np.array([50, 100, 200, 500, 1000])
y_means = np.array([0.72, 0.81, 0.88, 0.92, 0.95])
y_errs = np.array([0.05, 0.04, 0.03, 0.02, 0.01])

f, ax = cp.fig()
ax.scatter(x_data, y_means, yerr=y_errs, label='Model A', err_label='±1 SE')
ax.clean(xlabel='Training samples', ylabel='Accuracy')
f.savefig(FIGDIR / 'scatter_errorbars.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 3a2. Scatter with both x and y error bars ---
xy_x = np.array([10, 30, 60, 100, 150])
xy_y = 0.80 + 0.0010 * xy_x + np.random.normal(0, 0.008, len(xy_x))
xy_xerr = np.array([2, 5, 8, 10, 15])
xy_yerr = np.array([0.02, 0.018, 0.014, 0.012, 0.010])

f, ax = cp.fig()
ax.scatter(xy_x, xy_y, xerr=xy_xerr, yerr=xy_yerr, label='Model A',
           xerr_label='time uncertainty', yerr_label='accuracy uncertainty')
ax.clean(xlabel='Training time (min)', ylabel='Accuracy')
f.savefig(FIGDIR / 'scatter_xy_errorbars.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 3b. Matrix scatter (Nx2) — PCA/embedding ---
n_points = 200
cluster1 = np.random.randn(n_points, 2) * 0.5 + [2, 2]
cluster2 = np.random.randn(n_points, 2) * 0.5 + [-1, -1]
cluster3 = np.random.randn(n_points, 2) * 0.5 + [2, -1]

f, ax = cp.fig()
ax.scatter(cluster1, label='Class A', s=15, alpha=0.6)
ax.scatter(cluster2, label='Class B', s=15, alpha=0.6)
ax.scatter(cluster3, label='Class C', s=15, alpha=0.6)
ax.clean(xlabel='PC 1', ylabel='PC 2')
f.savefig(FIGDIR / 'scatter_clusters.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 3c. Scatter with size (Nx3) — bubble chart ---
data_nx3 = np.column_stack([
    np.random.rand(30) * 10,
    np.random.rand(30) * 100,
    np.random.rand(30) * 200 + 20,
])

f, ax = cp.fig()
ax.scatter(data_nx3, alpha=0.6)
ax.clean(xlabel='Learning rate (x10)', ylabel='Accuracy')
f.savefig(FIGDIR / 'scatter_bubble.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 4. HEATMAPS
# =============================================================================

# --- 4a. Confusion matrix ---
conf_matrix = pd.DataFrame(
    [[45, 3, 2], [4, 42, 4], [1, 5, 44]],
    index=['Cat', 'Dog', 'Bird'],
    columns=['Cat', 'Dog', 'Bird'],
)

f, ax = cp.fig()
ax.heatmap(conf_matrix, fmt='d', cmap='Blues')
ax.clean(ticks=False, spines=None, ylabel='True', xlabel='Predicted')
f.savefig(FIGDIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 4b. Correlation matrix ---
np.random.seed(42)
corr_data = np.random.randn(100, 5)
corr_data[:, 1] += corr_data[:, 0] * 0.8
corr_data[:, 3] -= corr_data[:, 2] * 0.5
corr_df = pd.DataFrame(corr_data, columns=['LR', 'Dropout', 'Batch', 'WD', 'Epochs'])
corr = corr_df.corr()

f, ax = cp.fig()
ax.heatmap(corr, cmap=cp.cmaps.diverging, fmt='.2f')
ax.clean(ticks=False, spines=None)
f.savefig(FIGDIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 4c. Hyperparameter grid ---
lrs = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64, 128]
grid = pd.DataFrame(
    np.random.uniform(0.85, 0.98, (len(lrs), len(batch_sizes))),
    index=[str(lr) for lr in lrs],
    columns=[str(bs) for bs in batch_sizes],
)

f, ax = cp.fig()
ax.heatmap(grid, cmap=cp.cmaps.sequential, fmt='.3f')
ax.clean(ticks=False, spines=None, ylabel='Learning rate', xlabel='Batch size')
f.savefig(FIGDIR / 'hyperparam_grid.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 5. BOX & VIOLIN PLOTS
# =============================================================================

# --- 5a. Box plot — model comparison ---
box_data = {
    'LR': np.random.normal(0.82, 0.03, 30),
    'RF': np.random.normal(0.91, 0.02, 30),
    'GB': np.random.normal(0.93, 0.015, 30),
    'MLP': np.random.normal(0.89, 0.04, 30),
}

f, ax = cp.fig()
ax.box(box_data)
ax.clean(ylabel='Accuracy')
f.savefig(FIGDIR / 'box_plot.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5b. Grouped box plot (pivot DataFrame) ---
box_pivot = pd.DataFrame({
    'Precision': [np.random.normal(0.90, 0.02, 20), np.random.normal(0.88, 0.03, 20)],
    'Recall': [np.random.normal(0.87, 0.03, 20), np.random.normal(0.92, 0.02, 20)],
}, index=['RF', 'MLP'])

f, ax = cp.fig()
ax.box(box_pivot)
ax.clean(ylabel='Score')
f.savefig(FIGDIR / 'grouped_box.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5c. Violin plot — model comparison ---
f, ax = cp.fig()
ax.violin(box_data)
ax.clean(ylabel='Accuracy')
f.savefig(FIGDIR / 'violin_plot.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5d. Grouped violin (pivot) ---
f, ax = cp.fig()
ax.violin(box_pivot)
ax.clean(ylabel='Score')
f.savefig(FIGDIR / 'grouped_violin.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5e. Strip plot — raw points instead of an aggregate summary ---
f, ax = cp.fig()
ax.strip(box_data)
ax.clean(ylabel='Accuracy')
f.savefig(FIGDIR / 'strip_plot.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5e2. Grouped strip plot (pivot DataFrame) ---
f, ax = cp.fig()
ax.strip(box_pivot)
ax.clean(ylabel='Score')
f.savefig(FIGDIR / 'grouped_strip.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5f. Overlaid histograms (shared bins) ---
baseline = np.random.normal(0.0, 1.0, 2000)
tuned    = np.random.normal(0.7, 0.8, 2000)
ablation = np.random.normal(-0.5, 1.3, 2000)

f, ax = cp.fig()
ax.hist([baseline, tuned, ablation],
        label=['Baseline', 'Tuned', 'Ablation'],
        bins=40, density=True)
ax.clean(xlabel='Prediction error', ylabel='Density')
f.savefig(FIGDIR / 'hist_overlay.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 5f2. Same thing, log-spaced bins ---
run_a = np.random.lognormal(0, 0.8, 2000)
run_b = np.random.lognormal(0.6, 0.6, 2000)

f, ax = cp.fig()
ax.hist([run_a, run_b], label=['Run A', 'Run B'], bins=40, log_x=True)
ax.clean(xlabel='Latency (ms)', ylabel='Count')
f.savefig(FIGDIR / 'hist_overlay_log.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 6. IMAGES & COLORMAPS
# =============================================================================

# --- 6a. Image with no axes ---
image = np.random.rand(28, 28)

f, ax = cp.fig()
ax.imshow(image, cmap=cp.cmaps.gray)
ax.clean(ticks=None, spines=None)
f.savefig(FIGDIR / 'image_display.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 6b. Color labels (legend alternative) ---
f, ax = cp.fig()
for i, name in enumerate(['MLP', 'RF', 'Transformer']):
    ax.line(np.sort(np.random.randn(50).cumsum()), color=cp.colors[i])
ax.color_labels(['MLP', 'RF', 'Transformer'], x=0.05, ha='left')
ax.clean(xlabel='Step', ylabel='Score')
f.savefig(FIGDIR / 'color_labels.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 6c. Log scale ---
x_log = np.array([10, 50, 100, 500, 1000, 5000])
y_log = 1 - 1.0 / x_log + np.random.normal(0, 0.01, len(x_log))

f, ax = cp.fig()
ax.line(x_log, y_log, marker='o')
ax.set_xscale('log')
ax.clean(xlabel='Training samples', ylabel='Accuracy')
f.savefig(FIGDIR / 'log_scale.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 7. MULTI-PANEL LAYOUTS
# =============================================================================

def _build_grid():
    f, axes = cp.fig(rows=2, cols=2)
    axes[0, 0].line(train_loss)
    axes[0, 0].clean(ylabel='Loss', xlabel='Epoch')
    axes[0, 1].bar(['A', 'B', 'C'], [0.9, 0.85, 0.95])
    axes[0, 1].clean(ylabel='Accuracy')
    axes[1, 0].scatter(np.random.randn(50, 2), s=20, alpha=0.6)
    axes[1, 0].clean(xlabel='PC1', ylabel='PC2')
    axes[1, 1].box({'M1': np.random.randn(30), 'M2': np.random.randn(30) + 0.5})
    axes[1, 1].clean(ylabel='Score')
    return f

# --- 7a. 2x2 grid ---
f = _build_grid()
f.tight_layout()
f.savefig(FIGDIR / 'grid_layout.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 7b. Same grid with extra subplots_adjust spacing ---
f = _build_grid()
f.subplots_adjust(hspace=0.55, wspace=0.45)
f.savefig(FIGDIR / 'grid_layout_spaced.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 8. HARD-TO-GOOGLE TRICKS
# =============================================================================

# --- 8a. Rotate tick labels (and align them to the tick) ---
long_labels = ['convolutional_net', 'random_forest', 'gradient_boost',
               'transformer_large', 'recurrent_net']
scores = [0.82, 0.91, 0.93, 0.89, 0.95]

f, (ax1, ax2) = cp.fig(cols=2)
ax1.bar(long_labels, scores)
ax1.clean(ylabel='Accuracy', title='default')

ax2.bar(long_labels, scores)
ax2.tick_params(axis='x', rotation=45)
plt.setp(ax2.get_xticklabels(), ha='right')
ax2.clean(ylabel='Accuracy', title="rotation=45, ha='right'")
f.tight_layout()
f.savefig(FIGDIR / 'trick_rotation.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 8d. annotate a point ---
x8d = np.array([10, 30, 60, 100, 150, 200])
y8d = np.array([0.70, 0.81, 0.87, 0.92, 0.93, 0.935])

f, ax = cp.fig()
ax.line(x8d, y8d, marker='o')
ax.annotate('best',
            xy=(150, 0.93), xytext=(100, 0.80),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.clean(xlabel='Training steps', ylabel='Accuracy')
f.savefig(FIGDIR / 'trick_annotate.png', dpi=150, bbox_inches='tight')
plt.close(f)

# --- 8e. set_aspect('equal') for scatter ---
theta = np.linspace(0, 2 * np.pi, 200)
cx = np.cos(theta) + np.random.normal(0, 0.03, len(theta))
cy = np.sin(theta) + np.random.normal(0, 0.03, len(theta))

f, (ax1, ax2) = cp.fig(cols=2)
ax1.scatter(cx, cy, s=10, alpha=0.6)
ax1.clean(xlabel='x', ylabel='y', title='default')

ax2.scatter(cx, cy, s=10, alpha=0.6)
ax2.set_aspect('equal')
ax2.clean(xlabel='x', ylabel='y', title="set_aspect('equal')")
f.tight_layout()
f.savefig(FIGDIR / 'trick_aspect_equal.png', dpi=150, bbox_inches='tight')
plt.close(f)


# =============================================================================
# 9. COLORMAP SWATCHES (for the Colormaps table in gallery.md)
# =============================================================================

cmap_names = ['inferno', 'plasma', 'viridis', 'magma',
              'RdBu', 'coolwarm', 'PiYG',
              'gray', 'categorical', 'phase']
gradient = np.linspace(0, 1, 256).reshape(1, -1)
for name in cmap_names:
    cmap = getattr(cp.cmaps, name)
    f, ax = plt.subplots(figsize=(2.5, 0.25))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(FIGDIR / f'cmap_{name}.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(f)


print(f"Generated {len(list(FIGDIR.glob('*.png')))} figures in {FIGDIR}")
