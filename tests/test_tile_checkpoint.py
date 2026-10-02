from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

import tile_checkpoint
from analysis_session import save_analysis_session
from tile_checkpoint import TileCheckpoint, checkpoint_lock, load_checkpoint_run
from test_analysis_view_restore import app, populate, restore_tk_default_root


def run_fixture():
    live = populate(app())
    live._worker_status = Mock()
    live._origami_identification_worker_progress = Mock()
    payload = dict(templates=[dict(name=name, **value) for name, value in live.origami_multi_template_results.items()],
                   counts=np.array([12, 12]), unclassified_count=12,
                   unclassified_centers_nm=live.origami_multi_template_unclassified_centers_nm,
                   unclassified_details=[])
    run = dict(checkpoint_version=1, run_id='test-run', analysis=live._capture_origami_analysis(),
               context=dict(source_locs=live.corrected_locs, pixelsize=1.,
                            tile_lattice=[(-50., 1800., -50., 50.), (1800., 4000., -50., 50.)],
                            tile_indices=np.array([0, 1]),
                            identification_params=dict(custom_templates=('A', 'Full'), identification_generation=7),
                            source_params={'source': 'Corrected localizations', 'active_filters': []},
                            overlay_params={'source_path': live.loaded.path}))
    return live, run, payload


def test_crash_after_first_tile_resumes_from_embedded_snapshot(tmp_path):
    live, run, payload = run_fixture()
    live._identify_origami_worker = Mock(side_effect=[('origami_multi_picks', deepcopy(payload)), RuntimeError('simulated crash')])
    with pytest.raises(RuntimeError, match='simulated crash'):
        live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    assert (tmp_path / 'run.paintanalysis').exists()
    assert (tmp_path / 'tile-00000000.paintanalysis').exists()
    assert not (tmp_path / 'tile-00000001.paintanalysis').exists()
    recovered = load_checkpoint_run(tmp_path, live._analysis_record_types())
    np.testing.assert_allclose(recovered['context']['source_locs'], live.corrected_locs)
    # A new app process can restore despite the original CSV not existing.
    resumed = populate(app())
    resumed._worker_status = Mock()
    resumed._origami_identification_worker_progress = Mock()
    resumed._identify_origami_worker = Mock(return_value=('origami_multi_picks', deepcopy(payload)))
    recovered['context']['identification_params']['identification_generation'] = 99
    kind, actual = resumed._checkpointed_tiled_worker(tmp_path, recovered)
    assert kind == 'origami_multi_picks'
    resumed._identify_origami_worker.assert_called_once()
    assert resumed._identify_origami_worker.call_args.args[0][:, 0].min() >= 1800.
    np.testing.assert_array_equal(actual['counts'], [24, 24])
    assert actual['unclassified_count'] == 24
    assert actual['candidate_count'] == 72
    assert all(item['params']['identification_generation'] == 99 for item in actual['templates'])
    # Once all tiles are saved, even a crash during aggregation loses no tile work.
    resumed._identify_origami_worker = Mock(side_effect=AssertionError('Recomputed a completed tile'))
    _, again = resumed._checkpointed_tiled_worker(tmp_path, recovered)
    np.testing.assert_array_equal(again['counts'], actual['counts'])
    for left, right in zip(actual['templates'], again['templates']):
        np.testing.assert_array_equal(left['picks'].accepted_mask, right['picks'].accepted_mask)
        for region_a, region_b in zip(left['picks'].aligned_regions, right['picks'].aligned_regions):
            np.testing.assert_allclose(region_a, region_b)


def test_incomplete_or_foreign_tiles_are_recomputed_and_disk_failure_preserves_prior_tiles(tmp_path, monkeypatch):
    store = TileCheckpoint(tmp_path, 'one', {}, Mock())
    store.save(0, (0, 1, 0, 1), 'single', {'core_count': 3})
    assert store.load(0, (0, 1, 0, 1), 'single') == {'core_count': 3}
    assert store.load(0, (0, 2, 0, 1), 'single') is None
    assert TileCheckpoint(tmp_path, 'two', {}).load(0, (0, 1, 0, 1), 'single') is None
    store.path(1).write_bytes(b'interrupted zip')
    assert store.load(1, (1, 2, 0, 1), 'single') is None
    before = store.path(0).read_bytes()
    monkeypatch.setattr(tile_checkpoint, 'save_analysis_session', Mock(side_effect=OSError(28, 'No space left')))
    with pytest.raises(OSError, match='Previously saved tiles are retained'):
        store.save(2, (2, 3, 0, 1), 'single', {})
    assert store.path(0).read_bytes() == before


def test_checkpoint_lock_prevents_two_writers_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError):
        with checkpoint_lock(tmp_path):
            with pytest.raises(ValueError, match='already in use'):
                with checkpoint_lock(tmp_path):
                    pass
            raise RuntimeError('crash')
    with checkpoint_lock(tmp_path):
        pass


def test_single_template_resume_skips_completed_detection(tmp_path, monkeypatch):
    import paint_analysis_gui as gui
    live, run, _ = run_fixture()
    picks = live.origami_pick_result
    params = dict(live.origami_identification_params,
                  pick_bin_size_nm=5., connect_distance_nm=20., density_threshold=.2,
                  alignment_pixel_nm=2., alignment_iterations=2)
    params.pop('custom_templates', None)
    run['context']['identification_params'] = params
    run['context']['overlay_params'].update(
        rows=2, columns=2, spacing_x_nm=10., spacing_y_nm=10., site_radius_nm=3.,
        g5m_sigma_min_nm=1., g5m_sigma_max_nm=8., g5m_min_locs=2, g5m_bic_patience=2,
        allow_mirror=False, overlay_pixel_nm=1., overlay_padding_nm=2., overlay_blur_nm=1.)
    detect = Mock(side_effect=[deepcopy(picks), RuntimeError('interrupted')])
    monkeypatch.setattr(gui, 'identify_origami_regions', detect)
    with pytest.raises(RuntimeError, match='interrupted'):
        live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    detect = Mock(return_value=deepcopy(picks))
    monkeypatch.setattr(gui, 'identify_origami_regions', detect)
    monkeypatch.setattr(gui, 'align_picked_origamis', Mock(return_value=object()))
    kind, result = live._checkpointed_tiled_worker(tmp_path, load_checkpoint_run(tmp_path, live._analysis_record_types()))
    assert kind == 'origami_tiled'
    detect.assert_called_once()
    assert result['picks'].accepted_count == 12
    assert result['candidate_count'] == 36


def test_resume_restores_saved_configuration_and_rebases_generation(tmp_path):
    live, run, _ = run_fixture()
    saved_generation = live.origami_identification_generation
    restored = populate(app())
    restored.origami_identification_generation = 100
    for name in ('origami_identify_button', 'origami_tiled_button', 'origami_n_tiles_button', 'origami_random_roi_button'):
        setattr(restored, name, Mock())
    restored._set_origami_step_progress = Mock()
    restored._remember_file_dialog_dir = Mock()
    restored._run_worker = Mock()
    restored._checkpointed_tiled_worker = Mock()
    restored._resume_tiled_checkpoint(tmp_path, run)
    assert restored.origami_identification_running
    assert run['context']['identification_params']['identification_generation'] == restored.origami_identification_generation
    assert restored.origami_identification_generation != saved_generation
    assert restored.origami_checkpoint_location.get() == str(tmp_path)
    restored._run_worker.call_args.args[0]()
    restored._checkpointed_tiled_worker.assert_called_once_with(tmp_path, run)
