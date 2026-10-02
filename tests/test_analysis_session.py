from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import tkinter as tk
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from analysis_session import save_analysis_session, load_analysis_session
from origami_analysis import OrigamiPickResult, OrigamiAnalysisResult, identify_origami_regions, align_picked_origamis
from paint_analysis_gui import LoadedData, PaintAnalysisApp


def fixture():
    points = np.random.default_rng(1).normal(size=(30, 2))
    picks = identify_origami_regions(points, pick_bin_size_nm=2, connect_distance_nm=5,
        density_threshold=0, min_candidate_points=1, max_candidate_points=100,
        rows=2, columns=2, spacing_x_nm=10, spacing_y_nm=10,
        measure_sites=False, compute_grid_blob_bic=False)
    picks = replace(picks, accepted_mask=np.ones(len(picks.regions), bool))
    locs = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'frame': np.arange(30, dtype=np.uint32)})
    locs.index = pd.Index(np.arange(30) * 2, name='source_row')
    loaded = LoadedData(Path('/missing/source.hdf5'), locs, [{'Pixelsize': 1}], {'note': 'test'})
    params = {'identification_generation': 42, 'mirror_all_templates': True, 'digital_pixel_model': {'bit_brightness_factors': (2., .2)},
              'classification_lookup_eligible': (True,), 'classification_dispositions': ('classified as A',),
              'digital_pixel_probabilities': ((.9, .1),), '_tiled_full_field_display': True}
    overlay = align_picked_origamis(picks.aligned_regions, rows=2, columns=2,
                                   spacing_x_nm=10, spacing_y_nm=10, site_radius_nm=5, prealigned=True, use_g5m=False)
    state = dict(loaded=loaded, corrected_locs=locs, origami_source_locs=locs,
        origami_source_points_nm=points, origami_pick_result=picks, origami_identification_params=params,
        origami_multi_template_results={'A': {'picks': picks, 'params': params}},
        origami_multi_template_counts={'A': 1}, origami_multi_template_unclassified_count=0,
        origami_multi_template_overlay_results={'A': {'result': overlay}},
        origami_result=overlay, origami_type_count_order=['Unclassified', 'A'],
        origami_shared_alignment_template={'image': np.eye(3), 'path': Path('/missing/template.png')})
    return dict(state=state, settings={'origami_mirror_all_templates': True, 'pixel_size_nm': 1.}, filter_enabled={})


RECORDS = {cls.__name__: cls for cls in (LoadedData, OrigamiPickResult, OrigamiAnalysisResult)}


def test_real_results_round_trip_without_source_files(tmp_path):
    payload = fixture()
    path = tmp_path / 'whole.paintanalysis'
    save_analysis_session(path, payload)
    restored = load_analysis_session(path, RECORDS)
    PaintAnalysisApp._validate_origami_analysis(None, restored)
    state = restored['state']
    pd.testing.assert_frame_equal(state['loaded'].locs, payload['state']['loaded'].locs)
    assert state['loaded'].locs is state['corrected_locs'] is state['origami_source_locs']
    assert state['origami_pick_result'] is state['origami_multi_template_results']['A']['picks']
    np.testing.assert_equal(state['origami_pick_result'].aligned_regions, payload['state']['origami_pick_result'].aligned_regions)
    np.testing.assert_equal(state['origami_result'].site_counts, payload['state']['origami_result'].site_counts)
    assert state['origami_identification_params'] == payload['state']['origami_identification_params']
    assert state['origami_type_count_order'] == ['Unclassified', 'A']
    assert state['loaded'].path == Path('/missing/source.hdf5')
    with ZipFile(path) as archive:
        assert all(name.endswith(('.npy', '.json')) for name in archive.namelist())


def test_atomic_failure_preserves_previous_save(tmp_path):
    path = tmp_path / 'result.paintanalysis'
    save_analysis_session(path, {'good': np.arange(3)})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        save_analysis_session(path, {'bad': object()})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_reject_unsupported_version_and_corrupt_arrays(tmp_path):
    path = tmp_path / 'bad.paintanalysis'
    with ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'format': 'paint-analysis-session', 'version': 999}))
    with pytest.raises(ValueError, match='version'):
        load_analysis_session(path, RECORDS)
    with ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'format': 'paint-analysis-session', 'version': 1,
            'root': {'ref': 0}, 'nodes': [{'kind': 'record', 'name': 'Executable', 'values': {}}]}))
    with pytest.raises(ValueError, match='record'):
        load_analysis_session(path, RECORDS)


def test_ragged_arrays_and_nonfinite_values(tmp_path):
    objects = np.empty(2, dtype=object)
    objects[0], objects[1] = np.zeros((2, 2)), np.ones((3, 2))
    path = tmp_path / 'ragged.paintanalysis'
    save_analysis_session(path, {'array': objects, 'score': (-np.inf, np.nan), 'ids': {1, 2}})
    restored = load_analysis_session(path, RECORDS)
    for original, saved in zip(objects, restored['array']):
        np.testing.assert_array_equal(original, saved)
    assert np.isnan(restored['score'][1]) and restored['ids'] == {1, 2}


def test_install_restores_settings_and_results_without_analysis(tmp_path):
    path = tmp_path / 'whole.paintanalysis'
    save_analysis_session(path, fixture())
    payload = load_analysis_session(path, RECORDS)
    interpreter = tk.Tcl()
    app = SimpleNamespace(
        _validate_origami_analysis=lambda p: PaintAnalysisApp._validate_origami_analysis(None, p),
        _after_load=Mock(), _refresh_filter_list=Mock(), _update_filter_bounds_label=Mock(),
        _update_roi_label=Mock(), _origami_identification_snapshot=lambda: ('restored',),
        _on_origami_identification_setting_changed=Mock(), _show_origami_stage=Mock(),
        _refresh_origami_action_states=Mock(), _finish_origami_identification_progress=Mock(), render_origami_plot=Mock(), notebook=Mock(),
        origami_identification_generation=0,
        origami_mirror_all_templates=tk.BooleanVar(interpreter, False),
        pixel_size_nm=tk.DoubleVar(interpreter, 100),
        origami_template_result_view=tk.StringVar(interpreter, ''),
        origami_plot_option=tk.StringVar(interpreter, ''),
        identify_origamis=Mock(), _identify_origami_worker=Mock())
    PaintAnalysisApp._install_origami_analysis(app, payload)
    assert app.origami_mirror_all_templates.get()
    assert app.pixel_size_nm.get() == 1
    assert app.origami_multi_template_counts == {'A': 1}
    assert app.origami_identification_params['identification_generation'] == 1
    assert app.origami_multi_template_results['A']['params']['identification_generation'] == 1
    assert app.origami_multi_template_overlay_results['A']['identification_generation'] == 1
    assert app.origami_multi_template_overlay_results['A']['result'].origami_count == 1
    app.identify_origamis.assert_not_called()
    app._identify_origami_worker.assert_not_called()
    app.render_origami_plot.assert_called_once()
    app._after_load.assert_called_once_with(payload['state']['loaded'], render_raw=False)
    payload['state']['origami_multi_template_counts']['A'] = 99
    app._after_load.reset_mock()
    with pytest.raises(ValueError, match='counts'):
        PaintAnalysisApp._install_origami_analysis(app, payload)
    app._after_load.assert_not_called()


def test_lazy_overlay_uses_current_session_id_not_saved_run_id():
    state = fixture()['state']
    app = SimpleNamespace(**state)
    app.origami_identification_generation = 1
    app.origami_loaded_source_path = Path('/missing/source.hdf5')
    app.origami_loaded_source_label = 'restored source'
    app.origami_template_result_view = SimpleNamespace(get=lambda: 'A')
    app.status = Mock()
    app._run_worker = lambda worker: worker()
    app._overlay_origami_worker = Mock()
    for name, value in dict(rows=2, columns=2, spacing_x_nm=10, spacing_y_nm=10,
                            g5m_sigma_min_nm=1, g5m_sigma_max_nm=8, g5m_min_locs=20,
                            g5m_bic_patience=3, site_radius_nm=5, allow_mirror=False,
                            overlay_pixel_nm=.5, overlay_padding_nm=20, overlay_blur_nm=1).items():
        setattr(app, 'origami_' + name, SimpleNamespace(get=lambda value=value: value))
    assert app.origami_identification_params['identification_generation'] == 42
    PaintAnalysisApp.overlay_origamis(app)
    args = app._overlay_origami_worker.call_args.args
    assert args[3]['identification_generation'] == 1
    assert args[3]['template_name'] == 'A'


def test_cached_overlay_is_installed_without_building_again():
    state = fixture()['state']
    app = SimpleNamespace(**state)
    app.origami_result = None
    app.origami_result_render_settings = None
    app.origami_identification_generation = 3
    app.origami_template_result_view = SimpleNamespace(get=lambda: 'A')
    app._plot_origami_analysis = Mock()
    app.overlay_origamis = Mock()
    PaintAnalysisApp._auto_build_active_template_overlay(app)
    app.overlay_origamis.assert_not_called()
    restored = app._plot_origami_analysis.call_args.args[0]
    assert restored['identification_generation'] == 3
    assert restored['result'] is state['origami_multi_template_overlay_results']['A']['result']
