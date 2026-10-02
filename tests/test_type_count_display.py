from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from paint_analysis_gui import PaintAnalysisApp


def test_custom_order_keeps_counts_markers_colors_and_labels_together():
    figure = Figure(figsize=(9, 5))
    canvas = FigureCanvasAgg(figure)
    app = SimpleNamespace(
        origami_multi_template_results={
            'closed': {'params': {'classification_raw_template_probabilities': [12.]}},
            'open': {'params': {'classification_raw_template_probabilities': [9.]}},
        },
        origami_multi_template_counts={'closed': 12, 'open': 7},
        origami_multi_template_unclassified_count=3,
        origami_type_count_order=['open', 'Unclassified', 'removed', 'closed'],
        origami_figure=figure, origami_canvas=canvas, origami_toolbar=Mock(),
        _configure_origami_navigation_controls=Mock(), notebook=Mock(), status=Mock())
    PaintAnalysisApp._plot_origami_type_counts(app)
    canvas.draw()
    axis = figure.axes[0]
    assert [t.get_text() for t in axis.get_xticklabels()] == ['open', 'Unclassified', 'closed']
    assert [bar.get_height() for bar in axis.patches] == [7, 3, 12]
    assert len(axis.collections) == 1  # Probability diamonds only; no expected mixture.
    np.testing.assert_allclose(axis.collections[0].get_offsets(), [[2, 12], [0, 9]])
    assert [text.get_text() for text in axis.texts] == ['7\nΣp=9.0', '3', '12']
    renderer = canvas.get_renderer()
    for text, top in zip(axis.texts, [9, 3, 12]):
        assert text.get_window_extent(renderer).y0 > axis.transData.transform((0, top))[1] + 7
    colors = {t.get_text(): bar.get_facecolor() for t, bar in zip(axis.get_xticklabels(), axis.patches)}
    app.origami_type_count_order = []
    PaintAnalysisApp._plot_origami_type_counts(app)
    for name, bar in zip(['closed', 'open', 'Unclassified'], figure.axes[0].patches):
        assert bar.get_facecolor() == colors[name]


def test_open_closed_percentages_pair_counts_and_follow_custom_order():
    counts = {'bit1_open': 30, 'bit1_closed': 70, 'bits1_4_closed': 1,
              'bits1_4_open': 199, 'bit2_off': 500, 'alone_open': 50,
              'empty_open': 0, 'empty_closed': 0, 'zero_open': 0, 'zero_closed': 10}
    figure = Figure(figsize=(10, 6))
    app = SimpleNamespace(
        origami_multi_template_results=dict.fromkeys(counts, {}),
        origami_multi_template_counts=counts,
        origami_multi_template_unclassified_count=999,
        origami_type_count_order=['bits1_4_closed', 'bit1_closed'],
        origami_figure=figure, origami_canvas=FigureCanvasAgg(figure),
        origami_toolbar=Mock(), _configure_origami_navigation_controls=Mock(),
        notebook=Mock(), status=Mock())
    PaintAnalysisApp._plot_open_vs_closed(app)
    app.origami_canvas.draw()
    axis = figure.axes[0]
    assert [t.get_text() for t in axis.get_xticklabels()] == ['bits1_4', 'bit1', 'empty', 'zero']
    np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                               [99.5, 30, 0, 0, .5, 70, 0, 100])
    np.testing.assert_allclose([bar.get_y() for bar in axis.patches[4:]], [99.5, 30, 0, 0])
    labels = [t.get_text() for t in axis.texts]
    assert 'n=200\nClosed 0.5%' in labels
    assert 'n=0 (no data)' in labels
    assert 'n=100' in labels
    assert app.origami_last_rendered_plot_option == 'Open vs closed'
    app.origami_multi_template_results = {'unpaired_open': {}}
    PaintAnalysisApp._plot_open_vs_closed(app)
    assert not figure.axes[0].patches
    assert 'No matching' in figure.axes[0].texts[0].get_text()


def test_open_closed_infers_aliases_without_merging_different_schemes():
    counts = {'bit2_open': 125, 'bit2_off': 500, 'bits1_4_ON': 50,
              'BITS1-4 closed': 30, 'bits1 4_close': 20,
              'bit1_closed': 15, 'bit10_open': 90, 'bit2_open_rigid': 900}
    figure = Figure(figsize=(10, 6))
    app = SimpleNamespace(
        origami_multi_template_results=dict.fromkeys(counts, {}),
        origami_multi_template_counts=counts, origami_type_count_order=[],
        origami_figure=figure, origami_canvas=FigureCanvasAgg(figure),
        origami_toolbar=Mock(), _configure_origami_navigation_controls=Mock(),
        notebook=Mock(), status=Mock())
    PaintAnalysisApp._plot_open_vs_closed(app)
    app.origami_canvas.draw()
    axis = figure.axes[0]
    assert [t.get_text() for t in axis.get_xticklabels()] == ['bit2', 'bits1_4']
    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], [20, 50, 80, 50])
    assert 'n=625' in [text.get_text() for text in axis.texts]
    assert 'Suffix aliases:' in axis.get_title()
    assert 'bit2_off → closed' in app.status.set.call_args.args[0]
