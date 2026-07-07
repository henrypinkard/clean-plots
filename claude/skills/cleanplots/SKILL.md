---
name: cleanplots
description: Make matplotlib plots with the cleanplots Python library (cp.fig() → plot calls → ax.clean()) — the user's standard for all Python plotting. Use whenever writing or editing Python plotting code for the user, whether paper figures, slide figures, notebooks, or quick data looks. Graph-type choice and figure design come from the jld skill's graphics bundle; this skill covers the library and the user's visual style.
---

# cleanplots — Method-Grade Plots in Python

`cleanplots` wraps matplotlib so high signal-to-noise defaults are the path of
least resistance. Repo: `~/GitRepos/clean-plots`; the full API reference with
worked examples is `docs/gallery.md` there — read the relevant section before
writing nontrivial plotting code.

## Workflow

```python
import cleanplots as cp
f, ax = cp.fig()                      # or cp.fig(rows=2, cols=2) for panels
ax.line(df, err=std, err_label='±1 SD')   # line / bar / barh / scatter / ...
ax.clean(xlabel='...', ylabel='...')      # always finish with clean()
```

Tidy data first: `cp.data.collapse(df, x, metrics, over='seed', func='mean')`
reshapes per-seed tidy frames into the wide form the plot calls take;
`cp.data.slice` filters by column level; `cp.data.to_wide` pivots raw runs.

## Choose the Graph with the Method

Load the jld skill's **graph** bundle when deciding what to plot:
`~/.claude/skills/jld/graphics/01_concepts.md` + `graphics/02_graphs.md`
(and the **caption** bundle, `graphics/03_captions.md`, for figures going into
documents). If the jld skill is not installed, the table below suffices.

| Question about the data | cleanplots call |
|---|---|
| Comparison among items | `ax.barh(series)` — horizontal, labels readable, bars from zero |
| Grouped comparison (items × metric) | `ax.barh(df, err=std_df)` — horizontal grouped bars (one group per column); vertical `ax.bar` only if the user asks (confirmed ruling) |
| Comparison of close values | dots on a scale that need not start at zero (position, not length) |
| Distribution | all points when practical; histogram; box plots only for large sets |
| Correlation (2 continuous vars) | `ax.scatter(df[['x', 'y']])` |
| Evolution over time | `ax.line` (mark the dots when data are sparse) |
| Subsets of a discrete variable | `group=` within one panel, or `cp.fig(rows, cols)` small multiples with shared scales |

Textbook rules that still bind here: bars drawn in full from a meaningful zero
(`zero_origin=True` when the range might not include it); never bars on a log
scale; no dual y-axes with different units (plot relative evolutions on one
scale, or juxtapose panels); log scales only to probe exponential/log relations
or compare rates of change — never just to squeeze a wide range.

## Trust the Library's Defaults

`clean()` already implements the quiet background: bottom-left spines only,
sparse ticks, no gridlines, and colored-text labels replacing the legend box.
Do not undo these, and do not add decoration back (3D, gradients, gridlines,
frames around fills).

## Where the User's Style Deviates from the Textbook (Deliberate)

- **Label placement.** The textbook wants labels directly adjacent to the
  data, and the user agrees — but adjacency is hard to automate. Accepted
  flow: place colored text near the data with
  `ax.color_labels([...], x=, y=)` when feasible; otherwise accept `clean()`'s
  automatic colored-text legend replacement and leave final adjacency for
  Illustrator. Don't contort code chasing perfect adjacency.
- **Tick density.** The textbook's two-tick, data-relevant scales are too
  sparse for the user's taste. Keep `clean()`'s `ticks='sparse'` default; do
  not strip axes down to two marks.

If some other textbook rule fights a library default or an existing figure,
flag the conflict to the user instead of silently picking a side; confirmed
rulings get added to this list.

## Captions

Every figure destined for a document gets a message-first, self-contained
caption (the takeaway as a complete sentence, then the minimum detail needed
to stand alone) — see the jld caption bundle above.
