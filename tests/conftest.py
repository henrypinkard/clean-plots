import matplotlib
matplotlib.use('Agg')

import pytest
import matplotlib.pyplot as plt
import numpy as np


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close('all')


@pytest.fixture
def numeric_axes():
    """Axes with a simple numeric line plot."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1.5, 3.7, 5])
    return fig, ax


@pytest.fixture
def bar_axes_with_strings():
    """Axes with a bar chart using string labels set via set_xticklabels (pandas-style)."""
    fig, ax = plt.subplots()
    labels = ['GB', 'KNN', 'LR', 'MLP', 'RF']
    values = [0.96, 0.98, 0.96, 0.94, 0.97]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    return fig, ax, labels


@pytest.fixture
def categorical_axes():
    """Axes with native matplotlib categorical bar chart."""
    fig, ax = plt.subplots()
    ax.bar(['A', 'B', 'C'], [1, 2, 3])
    return fig, ax


@pytest.fixture
def subplot_grid():
    """2x3 grid of axes, each with a line plot."""
    fig, axes = plt.subplots(2, 3)
    for ax in axes.flat:
        ax.plot([0, 1, 2], [0, 1, 2])
    return fig, axes
