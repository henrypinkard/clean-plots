---
name: cleanplots
description: Make matplotlib plots with the cleanplots Python library (cp.fig() → plot calls → ax.clean()) — the user's standard for all Python plotting. Use whenever writing or editing Python plotting code for the user: paper figures, slide figures, notebooks, quick data looks. Graph choice comes from the jld skill's graphics bundle; this skill covers the library and the user's visual style.
---

# cleanplots

Repo: `~/GitRepos/clean-plots`. API reference: `docs/gallery.md` there — read
the relevant section before writing nontrivial plotting code.

```python
import cleanplots as cp
f, ax = cp.fig()                      # or cp.fig(rows=2, cols=2) for panels
ax.line(df, err=std, err_label='±1 SD')   # line / bar / barh / scatter / ...
ax.clean(xlabel='...', ylabel='...')      # always finish with clean()
```

Tidy data first: `cp.data.collapse(df, x, metrics, over='seed', func='mean')`
reshapes per-seed frames into the wide form the plot calls take;
`cp.data.slice` filters by column level; `cp.data.to_wide` pivots raw runs.

## Choosing the Graph

Load the jld **graph** bundle: `~/.claude/skills/jld/graphics/01_concepts.md`
+ `graphics/02_graphs.md` (+ the **caption** bundle, `graphics/03_captions.md`,
for figures going into documents). If jld is absent, the table suffices.

| Question about the data | cleanplots call |
|---|---|
| Comparison among items | `ax.barh(series)` — horizontal, labels readable, bars from zero |
| Grouped comparison (items × metric) | `ax.barh(df)` — one group per column; vertical `ax.bar` only on request (user ruling) |
| Comparison with uncertainty | dots with two-sided whiskers: `ax.scatter(values, positions, xerr=err)`; bars with `err=` only on request (user ruling; textbook p. 134) |
| Comparison of close values | dots on a scale that need not start at zero |
| Distribution | all points when practical; histogram; box plots only for large sets |
| Correlation (2 continuous vars) | `ax.scatter(df[['x', 'y']])` |
| Evolution over time | `ax.line` (mark the dots when data are sparse) |
| Subsets of a discrete variable | `group=` within one panel, or `cp.fig(rows, cols)` small multiples with shared scales |

Binding rules: bars in full from a meaningful zero (`zero_origin=True` when
the range might not include it); never bars on a log scale; no dual y-axes
with different units (relative evolutions on one scale, or panels); log
scales only for exponential/log relations or comparing rates of change —
never just to squeeze a wide range.

## Defaults and Deviations

`clean()` implements the quiet background: bottom-left spines, sparse ticks,
no gridlines, colored-text labels instead of a legend box. Don't undo it;
don't add decoration (3D, gradients, gridlines, frames around fills).

Deliberate deviations from the textbook (user rulings):

- **Labels**: adjacency to data is right but hard to automate. Use
  `ax.color_labels([...], x=, y=)` when feasible; otherwise accept the
  colored-text legend — final adjacency happens in Illustrator.
- **Ticks**: keep `ticks='sparse'`; don't strip axes to the textbook's two marks.

If a textbook rule fights a library default or an existing figure, flag the
conflict to the user; rulings get added here.

## Captions

Figures going into documents get a message-first caption: takeaway as a
complete sentence, then the minimum to stand alone (jld **caption** bundle).
