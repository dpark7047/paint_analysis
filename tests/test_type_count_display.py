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
