"""Display regression checks; no localization files or Tk windows are opened."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from paint_analysis_gui import PaintAnalysisApp


def source_view():
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    image = np.arange(64., dtype=float).reshape(8, 8)
    app = SimpleNamespace(
        origami_figure=figure,
        origami_source_render_result=dict(
            image=image, extent=(0., 100., 0., 100.),
            min_density=0., max_density=63., blur_method="smooth",
        ),
    )
    PaintAnalysisApp._draw_origami_source_density(app, axis, np.empty((0, 2)))
    return app, axis, canvas


def test_repeated_on_off_overlays_do_not_emit_zoom_events():
    # Leave autoscaling enabled here to check the decision artist itself.
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.imshow(np.zeros((10, 10)))
    before = (axis.get_xlim(), axis.get_ylim())
    limits_changed = mock.Mock()
    axis.callbacks.connect("xlim_changed", limits_changed)
    axis.callbacks.connect("ylim_changed", limits_changed)
    polygon = np.array([[1., 1.], [2., 1.], [2., 2.]])
    blobs = [("ON", [polygon], "#22c55e", .18, "solid"),
             ("OFF", [polygon + 30.], "#ff3030", .14, "solid")]
    with mock.patch("paint_analysis_gui.digital_group_decision_blobs", return_value=blobs):
        for _ in range(4):
            artists = PaintAnalysisApp._draw_digital_decision_blobs(
                SimpleNamespace(), axis, np.empty((0, 2)), {}, 0,
            )
            canvas.draw()
            assert len(artists) == 2
            for artist in artists:
                artist.remove()
    assert (axis.get_xlim(), axis.get_ylim()) == before
    limits_changed.assert_not_called()


def test_source_viewport_stays_fixed_during_large_overlays_and_remains_zoomable():
    _, axis, canvas = source_view()
    limits_changed = mock.Mock()
    axis.callbacks.connect("xlim_changed", limits_changed)
    axis.callbacks.connect("ylim_changed", limits_changed)
    for _ in range(3):
        # Collections and scatter cover several diagnostic overlay styles,
        # including points outside the current viewport.
        collection = axis.add_collection(PolyCollection(
            [np.array([[200., 200.], [300., 200.], [300., 300.]])] * 500,
        ))
        points = axis.scatter([200.], [200.])
        canvas.draw()
        collection.remove()
        points.remove()
    assert axis.get_xlim() == (0., 100.)
    assert axis.get_ylim() == (0., 100.)
    limits_changed.assert_not_called()
    # A real user pan/zoom still triggers the app's refresh callbacks.
    axis.set_xlim(25., 50.)
    axis.set_ylim(25., 50.)
    assert limits_changed.call_count == 2
    assert axis.get_xlim() == (25., 50.)


def test_header_only_reflows_when_width_crosses_layout_boundary():
    names = (
        "origami_sidebar_toggle_button", "origami_view_label", "origami_plot_combo",
        "origami_template_result_label", "origami_template_result_combo",
        "origami_match_label", "origami_match_panel_combo", "origami_popout_button",
        "origami_fullscreen_button",
    )
    app = SimpleNamespace(**{name: mock.Mock() for name in names})
    for width in (1200, 1200, 1199, 1300):
        PaintAnalysisApp._layout_origami_plot_header(app, SimpleNamespace(width=width))
    assert app.origami_plot_combo.grid_configure.call_count == 1
    for width in (800, 800, 950):
        PaintAnalysisApp._layout_origami_plot_header(app, SimpleNamespace(width=width))
    assert app.origami_plot_combo.grid_configure.call_count == 2
    PaintAnalysisApp._layout_origami_plot_header(app, SimpleNamespace(width=1200))
    assert app.origami_plot_combo.grid_configure.call_count == 3
