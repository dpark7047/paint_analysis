from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from paint_analysis_gui import draw_digital_dark_space, origami_grid_points


def test_overlay_uses_search_mask_and_fitted_transform():
    params = dict(rows=3, columns=3, spacing_x_nm=10., spacing_y_nm=10.,
                  site_mask_radius_nm=2., alignment_template_image=np.ones((3, 3)),
                  digital_pixel_model=dict(physical_shape=(3, 3), bit_physical_cells=((0, 2), (6, 8)),
                       dark_groups=({"id": "empty", "cells": (4,), "mask_radius_nm": 3.},)))
    axis = Figure().subplots()
    rotation = np.array([[0., -1.], [1., 0.]])
    transform = lambda points: points @ rotation.T + [120., 80.]
    from paint_analysis_gui import _dark_overlay_paths
    artists = draw_digital_dark_space(axis, params, transform)
    assert len(artists) == 2
    expected = _dark_overlay_paths(((0., 0., 3.),))
    for shown, local in zip(artists[0].get_paths(), expected):
        np.testing.assert_allclose(np.linalg.norm(local.vertices, axis=1), 3., atol=.07)
        np.testing.assert_allclose(shown.vertices, transform(local.vertices))
        np.testing.assert_array_equal(shown.codes, local.codes)
    assert artists[0].get_hatch() == "///"
    for artist in artists:
        artist.remove()
    assert len(axis.collections) == 0


def test_no_overlay_without_alignment_gap_inputs():
    axis = Figure().subplots()
    assert draw_digital_dark_space(axis, {}) == []
    assert draw_digital_dark_space(axis, {'digital_pixel_model': {}}) == []
    assert not axis.collections


def test_failed_step3_fits_still_show_dark_mask_and_on_off_groups():
    from types import SimpleNamespace
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba
    from paint_analysis_gui import PaintAnalysisApp

    class Toggle:
        def __init__(self, value):
            self.value = value
        def get(self):
            return self.value

    grid = origami_grid_points(3, 3, 10., 10., {})
    corners = np.array([[[-15., -15.], [15., -15.], [15., 15.], [-15., 15.]]])
    model = dict(physical_shape=(3, 3), bit_ids=('A', 'B'),
                 bit_physical_cells=((0, 2), (6, 8)), active_bits=(True, True),
                 dark_groups=({'id': 'empty', 'cells': (4,), 'mask_radius_nm': 3.},))
    params = dict(_inspection_stage=4, template_mode='Custom image',
                  alignment_template_image=np.eye(3), rows=3, columns=3,
                  spacing_x_nm=10., spacing_y_nm=10., site_mask_radius_nm=2.,
                  digital_pixel_model=model, digital_pixel_probabilities=((1., 0.),),
                  min_site_localizations=5, min_site_evidence=.1)
    picks = SimpleNamespace(bounds_nm=np.array([[-15., 15., -15., 15.]]),
                            accepted_mask=np.array([False]), template_points_nm=grid,
                            rectangle_corners_nm=corners,
                            site_localization_counts=np.zeros((1, 9)),
                            site_prominence=np.zeros((1, 9)),
                            site_centroids_nm=np.full((1, 9, 2), np.nan))
    app = SimpleNamespace(origami_footprint_artists=[], origami_identification_params=params,
                          _active_origami_picks_and_params=lambda: (picks, params))
    for name in ('text_statistics', 'theoretical_overlay', 'dark_overlay', 'detected_sites_overlay',
                 'site_diagnostics', 'prominence_geometry'):
        setattr(app, 'origami_show_' + name, Toggle(name in {'theoretical_overlay', 'dark_overlay', 'site_diagnostics'}))
    axis = Figure().subplots()
    axis.set_xlim(-20, 20)
    axis.set_ylim(-20, 20)
    with patch('paint_analysis_gui.alignment_template_overlay_points', return_value=grid[[4]]):
        shown, total = PaintAnalysisApp._draw_visible_origami_footprints(app, axis, picks)
        assert (shown, total) == (1, 1)
        assert {label.get_text() for label in axis.get_legend().get_texts()} == {'Expected bright sites', 'Expected dark space'}
        assert any(artist.get_hatch() == '///'
                   for artist in axis.collections)
        colors = [artist.get_edgecolors()[0] for artist in axis.collections
                  if isinstance(artist, PolyCollection)]
        assert any(np.allclose(color, to_rgba('#22c55e', .95)) for color in colors)
        assert any(np.allclose(color, to_rgba('#ff3030', .95)) for color in colors)
        assert any('1 failed fits' in text.get_text() for text in axis.texts)
        count = len(axis.collections)
        PaintAnalysisApp._draw_visible_origami_footprints(app, axis, picks)
        assert len(axis.collections) == count  # Refresh removes prior overlays.
        for bright, dark, expected in (
            (False, True, {'Expected dark space'}),
            (True, False, {'Expected bright sites'}),
        ):
            app.origami_show_theoretical_overlay.value = bright
            app.origami_show_dark_overlay.value = dark
            PaintAnalysisApp._draw_visible_origami_footprints(app, axis, picks)
            assert {label.get_text() for label in axis.get_legend().get_texts()} == expected
            assert any(artist.get_hatch() == '///' for artist in axis.collections) == dark
    assert not picks.accepted_mask[0]  # Display does not accept the failed fit.


def test_dark_overlay_reuses_geometry_without_changing_view_limits():
    from paint_analysis_gui import _dark_overlay_paths
    params = dict(rows=3, columns=3, spacing_x_nm=10., spacing_y_nm=10.,
                  digital_pixel_model=dict(physical_shape=(3, 3), dark_groups=(
                      dict(id="dark", cells=(4,), mask_radius_nm=3.),)))
    _dark_overlay_paths.cache_clear()
    axis = Figure().subplots()
    axis.set_xlim(-20, 20)
    axis.set_ylim(-20, 20)
    changes = []
    axis.callbacks.connect('xlim_changed', lambda ax: changes.append('x'))
    axis.callbacks.connect('ylim_changed', lambda ax: changes.append('y'))
    for shift in (0., 100.):
        draw_digital_dark_space(axis, params, lambda points: points+shift)
    assert _dark_overlay_paths.cache_info().misses == 1
    assert _dark_overlay_paths.cache_info().hits == 1
    assert not changes
    np.testing.assert_equal(axis.get_xlim(), (-20, 20))
    np.testing.assert_equal(axis.get_ylim(), (-20, 20))
    params['digital_pixel_model']['dark_groups'][0]['mask_radius_nm'] = 4.
    draw_digital_dark_space(axis, params)
    assert _dark_overlay_paths.cache_info().misses == 2


def test_stage_defaults_enable_both_after_fit_and_only_bright_after_classification():
    from types import SimpleNamespace
    from paint_analysis_gui import PaintAnalysisApp

    class Toggle:
        def __init__(self):
            self.value = False
        def set(self, value):
            self.value = value
        def get(self):
            return self.value

    app = SimpleNamespace(**{'origami_show_' + name: Toggle() for name in (
        'theoretical_overlay', 'dark_overlay', 'alignment_overlay', 'detected_sites_overlay',
        'site_diagnostics', 'localization_group_assignments', 'prominence_geometry', 'text_statistics')})
    PaintAnalysisApp._set_origami_stage_preview_overlays(app, 2)
    assert app.origami_show_theoretical_overlay.get()
    assert app.origami_show_dark_overlay.get()
    PaintAnalysisApp._set_origami_stage_preview_overlays(app, 4)
    assert app.origami_show_theoretical_overlay.get()
    assert not app.origami_show_dark_overlay.get()
