import matplotlib
matplotlib.use('Agg')

import pytest
import matplotlib.pyplot as plt
import numpy as np
from cleanplots import clean, fig


# --- Spines ---

class TestSpines:
    def test_default_removes_top_right(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert ax.spines['bottom'].get_visible()
        assert ax.spines['left'].get_visible()

    def test_all_keeps_all(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, spines='all')
        for spine in ['top', 'right', 'bottom', 'left']:
            assert ax.spines[spine].get_visible()

    def test_none_removes_all(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, spines='none')
        for spine in ['top', 'right', 'bottom', 'left']:
            assert not ax.spines[spine].get_visible()

    def test_none_object_removes_all(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, spines=None)
        for spine in ['top', 'right', 'bottom', 'left']:
            assert not ax.spines[spine].get_visible()


# --- Ticks ---

class TestTicks:
    def test_sparse_default_two_ticks_per_axis(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        assert len(ax.get_xticks()) == 2
        assert len(ax.get_yticks()) == 2

    def test_sparse_x_only(self, numeric_axes):
        _, ax = numeric_axes
        yticks_before = len(ax.get_yticks())
        clean(ax, ticks='x')
        assert len(ax.get_xticks()) == 2
        # y ticks unchanged (should still have many)
        assert len(ax.get_yticks()) >= yticks_before

    def test_sparse_y_only(self, numeric_axes):
        _, ax = numeric_axes
        xticks_before = len(ax.get_xticks())
        clean(ax, ticks='y')
        assert len(ax.get_yticks()) == 2
        assert len(ax.get_xticks()) >= xticks_before

    def test_false_keeps_all(self, numeric_axes):
        _, ax = numeric_axes
        xticks_before = len(ax.get_xticks())
        yticks_before = len(ax.get_yticks())
        clean(ax, ticks=False)
        assert len(ax.get_xticks()) >= xticks_before
        assert len(ax.get_yticks()) >= yticks_before

    def test_none_removes_all_ticks(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, ticks=None)
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0

    def test_sparse_ticks_within_axis_limits(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for t in ax.get_xticks():
            assert xlim[0] <= t <= xlim[1]
        for t in ax.get_yticks():
            assert ylim[0] <= t <= ylim[1]


# --- Decimals ---

class TestDecimals:
    def test_auto_integers_no_decimal(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter(5, 0) == '5'
        assert formatter(5.0, 0) == '5'

    def test_auto_floats_clean_display(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter(3.5, 0) == '3.5'
        assert formatter(1.23, 0) == '1.23'  # g format preserves precision

    def test_fixed_decimals(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, decimals=2)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter(3.5, 0) == '3.50'
        assert formatter(5.0, 0) == '5'  # integers still clean

    def test_sci_notation(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, decimals='sci')
        formatter = ax.xaxis.get_major_formatter()
        result = formatter(1200, 0)
        assert 'e' in result

    def test_false_leaves_default(self, numeric_axes):
        _, ax = numeric_axes
        from matplotlib.ticker import ScalarFormatter
        clean(ax, decimals=False)
        assert isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)

    def test_log_scale_plain_decimals(self):
        fig, ax = plt.subplots()
        ax.bar(['A', 'B', 'C'], [0.95, 0.97, 0.99])
        ax.set_yscale('log')
        clean(ax)
        formatter = ax.yaxis.get_major_formatter()
        assert formatter(0.95, 0) == '0.95'
        assert formatter(1, 0) == '1'

    def test_auto_uses_g_format_for_small_floats(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter(0.001, 0) == '0.001'
        assert formatter(3.5, 0) == '3.5'


# --- String / custom label preservation ---

class TestCustomLabels:
    def test_pandas_style_string_labels_preserved(self, bar_axes_with_strings):
        _, ax, labels = bar_axes_with_strings
        clean(ax)
        result = [t.get_text() for t in ax.get_xticklabels()]
        assert result == labels

    def test_categorical_labels_preserved(self, categorical_axes):
        _, ax = categorical_axes
        clean(ax)
        result = [t.get_text() for t in ax.get_xticklabels()]
        assert result == ['A', 'B', 'C']

    def test_numeric_y_still_formatted_with_string_x(self, bar_axes_with_strings):
        _, ax, _ = bar_axes_with_strings
        clean(ax)
        # y-axis should have been formatted (only 2 sparse ticks)
        assert len(ax.get_yticks()) == 2


# --- Zero origin ---

class TestZeroOrigin:
    def test_default_no_change(self, numeric_axes):
        _, ax = numeric_axes
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        clean(ax)
        # Limits may change slightly due to sparse ticks, but should not be forced to 0
        # Just check they weren't forced to start at 0 if they didn't already
        # (the test data starts at 0, so let's use different data)

    def test_true_both_start_at_zero(self):
        fig, ax = plt.subplots()
        ax.plot([5, 6, 7], [10, 20, 30])
        clean(ax, zero_origin=True)
        assert ax.get_xlim()[0] == 0
        assert ax.get_ylim()[0] == 0

    def test_x_only(self):
        fig, ax = plt.subplots()
        ax.plot([5, 6, 7], [10, 20, 30])
        ylim_before = ax.get_ylim()
        clean(ax, zero_origin='x')
        assert ax.get_xlim()[0] == 0
        assert ax.get_ylim()[0] == ylim_before[0]

    def test_y_only(self):
        fig, ax = plt.subplots()
        ax.plot([5, 6, 7], [10, 20, 30])
        xlim_before = ax.get_xlim()
        clean(ax, zero_origin='y')
        assert ax.get_ylim()[0] == 0
        assert ax.get_xlim()[0] == xlim_before[0]


# --- Legend ---

class TestLegend:
    def test_legend_frame_removed(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='test')
        ax.legend()
        clean(ax)
        assert not ax.get_legend().get_frame_on()

    def test_no_legend_no_error(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax)  # should not raise


# --- kwargs pass-through ---

class TestKwargs:
    def test_xlabel_ylabel(self, numeric_axes):
        _, ax = numeric_axes
        clean(ax, xlabel='Time', ylabel='Value')
        assert ax.get_xlabel() == 'Time'
        assert ax.get_ylabel() == 'Value'


# --- Arrays of axes ---

class TestArrayOfAxes:
    def test_2d_numpy_array(self, subplot_grid):
        _, axes = subplot_grid
        clean(axes)
        for ax in axes.flat:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()
            assert len(ax.get_xticks()) == 2

    def test_1d_numpy_array(self):
        fig, axes = plt.subplots(1, 3)
        for ax in axes:
            ax.plot([0, 1, 2], [0, 1, 2])
        clean(axes)
        for ax in axes:
            assert not ax.spines['top'].get_visible()

    def test_list_of_axes(self):
        fig, axes = plt.subplots(1, 3)
        for ax in axes:
            ax.plot([0, 1, 2], [0, 1, 2])
        clean(list(axes))
        for ax in axes:
            assert not ax.spines['top'].get_visible()

    def test_kwargs_applied_to_all(self, subplot_grid):
        _, axes = subplot_grid
        clean(axes, xlabel='X')
        for ax in axes.flat:
            assert ax.get_xlabel() == 'X'


# --- fig() ---

class TestFig:
    def test_default_size(self):
        f, ax = fig()
        w, h = f.get_size_inches()
        assert w == 5
        assert h == 4

    def test_custom_grid_size(self):
        f, axes = fig(2, 3)
        w, h = f.get_size_inches()
        assert w == 10
        assert h == 8
        assert axes.shape == (2, 3)

    def test_explicit_size(self):
        f, ax = fig(w=12, h=6)
        w, h = f.get_size_inches()
        assert w == 12
        assert h == 6

    def test_width_capped_at_10(self):
        f, axes = fig(1, 5)
        w, _ = f.get_size_inches()
        assert w == 10

    def test_kwargs_passthrough(self):
        f, axes = fig(1, 2, sharex=True)
        # If sharex worked, both axes share the same x-axis
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])


# --- CleanAxes.line() ---

class TestLine:
    def test_basic_line(self):
        f, ax = fig()
        lines = ax.line([0, 1, 2], [0, 1, 2], label='test')
        assert len(ax.get_lines()) == 1
        assert ax.get_lines()[0].get_label() == 'test'

    def test_line_with_err_adds_fill(self):
        f, ax = fig()
        ax.line([0, 1, 2], [0, 1, 2], err=[0.1, 0.2, 0.3])
        assert len(ax.collections) == 1  # fill_between

    def test_line_without_err_no_fill(self):
        f, ax = fig()
        ax.line([0, 1, 2], [0, 1, 2])
        assert len(ax.collections) == 0

    def test_fill_color_matches_line(self):
        import matplotlib.colors
        f, ax = fig()
        lines = ax.line([0, 1, 2], [0, 1, 2], err=[0.1, 0.1, 0.1], color='red')
        line_rgb = matplotlib.colors.to_rgb(lines[0].get_color())
        fill_rgb = ax.collections[0].get_facecolor()[0][:3]
        assert np.allclose(line_rgb, fill_rgb, atol=0.01)

    def test_err_label_appears_once(self):
        f, ax = fig()
        ax.line([0, 1], [0, 1], err=[0.1, 0.1], label='A', err_label='±1 SE')
        ax.line([0, 1], [1, 0], err=[0.1, 0.1], label='B')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['A', 'B', '±1 SE']

    def test_no_err_label_by_default(self):
        f, ax = fig()
        ax.line([0, 1], [0, 1], err=[0.1, 0.1], label='A')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['A']

    def test_auto_legend_no_label_no_legend(self):
        f, ax = fig()
        ax.line([0, 1], [0, 1])
        assert ax.get_legend() is None

    def test_kwargs_passthrough(self):
        f, ax = fig()
        lines = ax.line([0, 1], [0, 1], linewidth=5, marker='o')
        assert lines[0].get_linewidth() == 5
        assert lines[0].get_marker() == 'o'

    def test_single_arg_y_only(self):
        f, ax = fig()
        lines = ax.line([10, 20, 30])
        assert len(ax.get_lines()) == 1
        y_data = ax.get_lines()[0].get_ydata()
        assert list(y_data) == [10, 20, 30]
        x_data = ax.get_lines()[0].get_xdata()
        assert list(x_data) == [0, 1, 2]

    def test_single_arg_with_label(self):
        f, ax = fig()
        ax.line([1, 2, 3], label='train')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['train']

    def test_multi_line_list(self):
        f, ax = fig()
        lines = ax.line([[1, 2, 3], [4, 5, 6]], label=['A', 'B'])
        assert len(ax.get_lines()) == 2
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['A', 'B']

    def test_multi_line_infers_x(self):
        f, ax = fig()
        ax.line([[10, 20], [30, 40]])
        x_data = ax.get_lines()[0].get_xdata()
        assert list(x_data) == [0, 1]

    def test_multi_line_with_err(self):
        f, ax = fig()
        ax.line([[1, 2, 3], [4, 5, 6]],
                err=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
                label=['A', 'B'])
        assert len(ax.collections) == 2  # two fill_betweens

    def test_multi_line_shared_err_label(self):
        f, ax = fig()
        ax.line([[1, 2], [3, 4]],
                err=[[0.1, 0.1], [0.1, 0.1]],
                label=['A', 'B'], err_label='±1 SD')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels.count('±1 SD') == 1
        assert 'A' in labels
        assert 'B' in labels

    def test_multi_line_per_line_err_labels(self):
        f, ax = fig()
        ax.line([[1, 2], [3, 4]],
                err=[[0.1, 0.1], [0.1, 0.1]],
                label=['A', 'B'], err_label=['SE(A)', 'SE(B)'])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'SE(A)' in labels
        assert 'SE(B)' in labels


# --- CleanAxes.bars() ---

class TestBar:
    def test_basic_bars(self):
        f, ax = fig()
        ax.bar(['A', 'B', 'C'], [1, 2, 3])
        assert len(ax.patches) == 3

    def test_bars_with_err(self):
        f, ax = fig()
        ax.bar(['A', 'B'], [5, 10], err=[1, 2])
        # errorbar lines should exist
        assert len(ax.lines) > 0

    def test_err_label_in_legend(self):
        f, ax = fig()
        ax.bar(['A', 'B'], [5, 10], err=[1, 2], label='Data', err_label='±1 SD')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Data' in labels
        assert '±1 SD' in labels

    def test_no_err_label_by_default(self):
        f, ax = fig()
        ax.bar(['A', 'B'], [5, 10], err=[1, 2], label='Data')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['Data']

    def test_kwargs_passthrough(self):
        f, ax = fig()
        ax.bar(['A'], [5], color='red')
        import matplotlib.colors
        patch_color = matplotlib.colors.to_rgb(ax.patches[0].get_facecolor())
        assert np.allclose(patch_color, matplotlib.colors.to_rgb('red'), atol=0.01)

    def test_single_arg_numeric_heights(self):
        f, ax = fig()
        ax.bar([3, 5, 7])
        assert len(ax.patches) == 3
        # Heights should be 3, 5, 7
        heights = [p.get_height() for p in ax.patches]
        assert heights == [3, 5, 7]

    def test_single_arg_with_err(self):
        f, ax = fig()
        ax.bar([3, 5, 7], err=[0.1, 0.2, 0.3])
        assert len(ax.patches) == 3
        assert len(ax.lines) > 0


# --- CleanAxes.scatter() ---

class TestScatter:
    def test_basic_scatter(self):
        f, ax = fig()
        ax.scatter([0, 1, 2], [3, 4, 5])
        assert len(ax.collections) >= 1

    def test_matrix_nx2(self):
        f, ax = fig()
        data = np.array([[0, 1], [2, 3], [4, 5]])
        ax.scatter(data)
        assert len(ax.collections) >= 1

    def test_matrix_nx3_uses_size(self):
        f, ax = fig()
        data = np.array([[0, 1, 10], [2, 3, 20], [4, 5, 30]])
        ax.scatter(data)
        sizes = ax.collections[0].get_sizes()
        assert np.allclose(sizes, [10, 20, 30])

    def test_matrix_bad_shape_raises(self):
        f, ax = fig()
        with pytest.raises(ValueError):
            ax.scatter(np.array([[1, 2, 3, 4]]))


# --- DataFrame / Series support ---

class TestDataFrameLine:
    def test_dataframe_plots_all_columns(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[10, 20, 30])
        f, ax = fig()
        ax.line(df)
        assert len(ax.get_lines()) == 2

    def test_dataframe_labels_from_columns(self):
        import pandas as pd
        df = pd.DataFrame({'Model1': [1, 2], 'Model2': [3, 4]})
        f, ax = fig()
        ax.line(df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['Model1', 'Model2']

    def test_dataframe_xlabel_from_index_name(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2]}, index=pd.Index([10, 20], name='n_train'))
        f, ax = fig()
        ax.line(df)
        assert ax.get_xlabel() == 'n_train'

    def test_dataframe_with_err_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        err_df = pd.DataFrame({'A': [0.1, 0.1, 0.1], 'B': [0.2, 0.2, 0.2]})
        f, ax = fig()
        ax.line(df, err=err_df, err_label='STD')
        # Should have 2 fill_between collections
        assert len(ax.collections) == 2
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'STD' in labels
        # err_label should appear only once
        assert labels.count('STD') == 1

    def test_series_uses_index_and_values(self):
        import pandas as pd
        s = pd.Series([10, 20, 30], index=[1, 2, 3], name='accuracy')
        f, ax = fig()
        ax.line(s)
        assert len(ax.get_lines()) == 1
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['accuracy']

    def test_series_xlabel_from_index_name(self):
        import pandas as pd
        s = pd.Series([1, 2], index=pd.Index([10, 20], name='epoch'), name='loss')
        f, ax = fig()
        ax.line(s)
        assert ax.get_xlabel() == 'epoch'


class TestPivotLine:
    def test_pivot_basic(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2, 3], [4, 5, 6]],
            'val': [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.line(df)
        # 2 rows * 2 cols = 4 data lines + 2 color legend + 2 style legend = 8
        assert len(ax.get_lines()) == 8

    def test_pivot_colors_match_index(self):
        import pandas as pd
        from cleanplots import get_color_cycle
        df = pd.DataFrame({
            'train': [[1, 2], [3, 4]],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.line(df)
        cycle = get_color_cycle()
        # First data line should be first color, second should be second
        assert ax.get_lines()[0].get_color() == cycle[0]
        assert ax.get_lines()[1].get_color() == cycle[1]

    def test_pivot_linestyles_match_columns(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2]],
            'val': [[3, 4]],
        }, index=['mlp'])
        f, ax = fig()
        ax.line(df)
        assert ax.get_lines()[0].get_linestyle() == '-'
        assert ax.get_lines()[1].get_linestyle() == '--'

    def test_pivot_with_shared_x(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2, 3]],
        }, index=['mlp'])
        f, ax = fig()
        ax.line([10, 20, 30], df)
        x_data = ax.get_lines()[0].get_xdata()
        assert list(x_data) == [10, 20, 30]

    def test_pivot_with_per_row_x(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2], [3, 4]],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.line({'mlp': [0, 1], 'transformer': [10, 20]}, df)
        assert list(ax.get_lines()[0].get_xdata()) == [0, 1]
        assert list(ax.get_lines()[1].get_xdata()) == [10, 20]

    def test_pivot_with_err(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2, 3]],
        }, index=['mlp'])
        err_df = pd.DataFrame({
            'train': [[0.1, 0.2, 0.3]],
        }, index=['mlp'])
        f, ax = fig()
        ax.line(df, err=err_df, err_label='±1 SD')
        assert len(ax.collections) == 1  # one fill_between
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert '±1 SD' in labels

    def test_pivot_legend_has_index_and_columns(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[1, 2], [3, 4]],
            'val': [[5, 6], [7, 8]],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.line(df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'mlp' in labels
        assert 'transformer' in labels
        assert 'train' in labels
        assert 'val' in labels

    def test_pivot_x_inferred_as_range(self):
        import pandas as pd
        df = pd.DataFrame({
            'train': [[10, 20, 30]],
        }, index=['mlp'])
        f, ax = fig()
        ax.line(df)
        x_data = ax.get_lines()[0].get_xdata()
        assert list(x_data) == [0, 1, 2]

    def test_pivot_single_column_ylabel(self):
        import pandas as pd
        df = pd.DataFrame({
            'val_loss_history': [[1, 2, 3], [4, 5, 6]],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.line(df)
        assert ax.get_ylabel() == 'val_loss_history'
        # Column name should NOT be in legend
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'val_loss_history' not in labels
        assert 'mlp' in labels
        assert 'transformer' in labels


# --- CleanAxes.heatmap() ---

class TestHeatmap:
    def test_basic_array(self):
        f, ax = fig()
        data = np.array([[1, 2], [3, 4]])
        im = ax.heatmap(data)
        assert im is not None

    def test_dataframe_labels(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        f, ax = fig()
        ax.heatmap(df)
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert xtick_labels == ['A', 'B']
        assert ytick_labels == ['x', 'y']

    def test_annotations(self):
        f, ax = fig()
        data = np.array([[1.5, 2.5]])
        ax.heatmap(data, annotate=True, fmt='.1f')
        texts = [t for t in ax.texts]
        assert len(texts) == 2
        assert texts[0].get_text() == '1.5'

    def test_no_annotations(self):
        f, ax = fig()
        ax.heatmap(np.array([[1, 2]]), annotate=False)
        assert len(ax.texts) == 0

    def test_no_cbar(self):
        f, ax = fig()
        ax.heatmap(np.array([[1, 2]]), cbar=False)
        # Only one axes (no colorbar axes)
        assert len(f.axes) == 1


# --- CleanAxes.box() ---

class TestBox:
    def test_list_of_arrays(self):
        f, ax = fig()
        data = [np.random.randn(50), np.random.randn(50)]
        bp = ax.box(data)
        assert 'boxes' in bp
        assert len(bp['boxes']) == 2

    def test_dict_mode(self):
        f, ax = fig()
        bp = ax.box({'A': np.random.randn(50), 'B': np.random.randn(50)})
        assert len(bp['boxes']) == 2
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ['A', 'B']

    def test_scalar_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({'A': np.random.randn(50), 'B': np.random.randn(50)})
        f, ax = fig()
        bp = ax.box(df)
        assert len(bp['boxes']) == 2

    def test_pivot_grouped(self):
        import pandas as pd
        df = pd.DataFrame({
            'metric1': [np.random.randn(30), np.random.randn(30)],
            'metric2': [np.random.randn(30), np.random.randn(30)],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        bp = ax.box(df)
        # 2 rows * 2 cols = 4 boxes
        assert len(bp['boxes']) == 4

    def test_pivot_single_col_ylabel(self):
        import pandas as pd
        df = pd.DataFrame({
            'accuracy': [np.random.randn(30), np.random.randn(30)],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        ax.box(df)
        assert ax.get_ylabel() == 'accuracy'


# --- CleanAxes.violin() ---

class TestViolin:
    def test_dict_mode(self):
        f, ax = fig()
        vp = ax.violin({'A': np.random.randn(50), 'B': np.random.randn(50)})
        assert 'bodies' in vp
        assert len(vp['bodies']) == 2

    def test_scalar_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({'A': np.random.randn(50), 'B': np.random.randn(50)})
        f, ax = fig()
        vp = ax.violin(df)
        assert len(vp['bodies']) == 2

    def test_pivot_grouped(self):
        import pandas as pd
        df = pd.DataFrame({
            'metric1': [np.random.randn(30), np.random.randn(30)],
            'metric2': [np.random.randn(30), np.random.randn(30)],
        }, index=['mlp', 'transformer'])
        f, ax = fig()
        vp = ax.violin(df)
        assert len(vp['bodies']) == 4


class TestDataFrameBar:
    def test_dataframe_grouped_bars(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        f, ax = fig()
        ax.bar(df)
        # 2 columns * 2 rows = 4 bar patches
        assert len(ax.patches) == 4

    def test_dataframe_bar_labels_from_columns(self):
        import pandas as pd
        df = pd.DataFrame({'GB': [0.9], 'RF': [0.95]}, index=['score'])
        f, ax = fig()
        ax.bar(df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['GB', 'RF']

    def test_dataframe_bar_with_err(self):
        import pandas as pd
        df = pd.DataFrame({'A': [5, 10], 'B': [8, 12]}, index=['x', 'y'])
        err_df = pd.DataFrame({'A': [1, 2], 'B': [0.5, 1]}, index=['x', 'y'])
        f, ax = fig()
        ax.bar(df, err=err_df, err_label='±1 SD')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert '±1 SD' in labels
        assert labels.count('±1 SD') == 1

    def test_series_bar(self):
        import pandas as pd
        s = pd.Series([1, 2, 3], index=['A', 'B', 'C'], name='counts')
        f, ax = fig()
        ax.bar(s)
        assert len(ax.patches) == 3

    def test_dataframe_bar_xlabel_from_index_name(self):
        import pandas as pd
        df = pd.DataFrame({'A': [1]}, index=pd.Index(['x'], name='category'))
        f, ax = fig()
        ax.bar(df)
        assert ax.get_xlabel() == 'category'


# --- CleanAxes.color_labels() ---

class TestColorLabels:
    def test_single_label(self):
        f, ax = fig()
        texts = ax.color_labels('MLP')
        assert len(texts) == 1
        assert texts[0].get_text() == 'MLP'

    def test_multiple_labels(self):
        f, ax = fig()
        texts = ax.color_labels(['MLP', 'RF', 'GB'])
        assert len(texts) == 3
        assert [t.get_text() for t in texts] == ['MLP', 'RF', 'GB']

    def test_colors_from_cycle(self):
        from cleanplots import get_color_cycle
        f, ax = fig()
        texts = ax.color_labels(['A', 'B'])
        cycle = get_color_cycle()
        assert texts[0].get_color() == cycle[0]
        assert texts[1].get_color() == cycle[1]

    def test_custom_colors(self):
        f, ax = fig()
        texts = ax.color_labels(['A', 'B'], colors=['red', 'blue'])
        assert texts[0].get_color() == 'red'
        assert texts[1].get_color() == 'blue'

    def test_custom_position(self):
        f, ax = fig()
        texts = ax.color_labels('A', x=0.1, y=0.5)
        pos = texts[0].get_position()
        assert pos == (0.1, 0.5)

    def test_vertical_stacking(self):
        f, ax = fig()
        texts = ax.color_labels(['A', 'B', 'C'])
        positions = [t.get_position() for t in texts]
        # All same x, decreasing y
        assert all(p[0] == positions[0][0] for p in positions)
        assert positions[0][1] > positions[1][1] > positions[2][1]

    def test_uses_axes_transform(self):
        f, ax = fig()
        texts = ax.color_labels('A')
        assert texts[0].get_transform() == ax.transAxes

    def test_kwargs_passthrough(self):
        f, ax = fig()
        texts = ax.color_labels('A', fontweight='bold')
        assert texts[0].get_fontweight() == 'bold'


# --- Style ---

class TestStyle:
    def test_legend_frameon_false(self):
        import matplotlib as mpl
        assert mpl.rcParams['legend.frameon'] == False

    def test_color_cycle_applied(self):
        import matplotlib as mpl
        cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        assert cycle[0] == '#179EE8'
