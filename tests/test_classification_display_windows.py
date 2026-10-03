"""Opt-in Windows Tk check using synthetic display data only."""
import os
import sys

import numpy as np
import pytest

from paint_analysis_gui import ORIGAMI_TAB, PaintAnalysisApp


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("PAINT_TEST_WINDOWS_GUI") != "1",
    reason="Set PAINT_TEST_WINDOWS_GUI=1 to run the transparent Windows Tk test",
)
def test_maximized_on_off_display_scrolls_and_reaches_step_four(monkeypatch):
    monkeypatch.setenv("PAINT_ANALYSIS_DISABLE_DEVELOPMENT_CACHE", "1")
    app = PaintAnalysisApp()
    app.attributes("-alpha", 0)
    app.notebook.select(ORIGAMI_TAB)
    app._show_origami_stage("Identify")
    app.state("zoomed")
    result = dict(limit_events=0, errors=[], button_responded=False)
    app.report_callback_exception = lambda _type, exc, _tb: result["errors"].append(str(exc))

    def changed(*_):
        result["limit_events"] += 1
        app._on_origami_view_limits_changed()

    def plot():
        result["initial_geometry"] = app.geometry()
        app.origami_figure.clear()
        axis = app.origami_figure.add_subplot(111)
        app.origami_source_render_result = dict(
            image=np.zeros((512, 512)), min_density=0., max_density=1.,
            extent=(0., 4000., 0., 4000.), blur_method="smooth",
        )
        app._draw_origami_source_density(axis, np.empty((0, 2)))
        axis.callbacks.connect("xlim_changed", changed)
        axis.callbacks.connect("ylim_changed", changed)
        grid = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        params = dict(
            digital_pixel_model=dict(bit_ids=("ON", "OFF"), bit_cells=((0, 1), (2, 3)), physical_shape=(2, 2)),
            digital_pixel_probabilities=((1., 0.),), site_mask_radius_nm=3.,
        )
        for index in range(500):
            shift = np.array([20. + index % 25 * 150, 20. + index // 25 * 150])
            app._draw_digital_decision_blobs(axis, grid, params, 0, lambda p, shift=shift: p + shift)
        app.origami_last_rendered_plot_option = "Identified origami template matches"
        app.origami_canvas.draw_idle()
        app.after(800, scroll)

    def scroll():
        canvas = app.origami_settings_canvas
        canvas.yview_moveto(0)
        result["scroll_before"] = canvas.yview()[0]
        canvas.event_generate("<MouseWheel>", delta=-120)
        app.after(100, reach_button)

    def reach_button():
        canvas = app.origami_settings_canvas
        result["scroll_after"] = canvas.yview()[0]
        button = app.origami_run_acceptance_button
        offset = button.winfo_rooty() - canvas.winfo_rooty() + canvas.canvasy(0)
        canvas.yview_moveto(max(0, offset - 100) / canvas.bbox("all")[3])
        button.state(["!disabled"])
        button.configure(command=lambda: result.update(button_responded=True))
        app.after(100, finish)

    def finish():
        canvas = app.origami_settings_canvas
        button = app.origami_run_acceptance_button
        result["button_visible"] = canvas.winfo_rooty() <= button.winfo_rooty() < canvas.winfo_rooty() + canvas.winfo_height()
        button.invoke()  # Exercise the control without running scientific analysis.
        result["final_geometry"] = app.geometry()
        result["window_state"] = app.state()
        app.quit()

    app.after(800, plot)
    app.after(12000, app.quit)
    try:
        app.mainloop()
        assert not result["errors"]
        assert result["limit_events"] == 0
        assert result["scroll_after"] > result["scroll_before"]
        assert result["button_visible"] and result["button_responded"]
        assert result["initial_geometry"] == result["final_geometry"]
        assert result["window_state"] == "zoomed"
    finally:
        app.destroy()
