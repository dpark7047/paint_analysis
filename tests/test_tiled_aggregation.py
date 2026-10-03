from copy import deepcopy
from dataclasses import fields
import gc
import weakref

import numpy as np
import pytest

from analysis_session import load_analysis_session, save_analysis_session
from origami_analysis import concatenate_origami_pick_results
from tiled_aggregation import StreamingTileResults, SEQUENCE_PARAMS, _Rows
from test_tile_checkpoint import run_fixture


def equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right or (isinstance(left, float) and np.isnan(left) and np.isnan(right))


def test_streamed_results_match_in_memory_merge_and_save_load(tmp_path):
    live, _, payload = run_fixture()
    second = deepcopy(payload)
    for item in second['templates']:
        item['picks'].density_extent_nm = (10., 200., -100., 40.)
        item['picks'].alignment_candidate_images *= 2
    writer = StreamingTileResults(tmp_path / 'cache', multi=True)
    writer.append(payload)
    writer.append(second)
    actual = writer.finish()
    np.testing.assert_equal(actual['counts'], payload['counts'] * 2)
    assert actual['unclassified_count'] == 24
    for index, item in enumerate(actual['templates']):
        expected = concatenate_origami_pick_results([payload['templates'][index]['picks'], second['templates'][index]['picks']])
        for field in fields(expected):
            equal(getattr(expected, field.name), getattr(item['picks'], field.name))
        assert isinstance(item['picks'].alignment_candidate_images, np.memmap)
        assert not item['picks'].alignment_candidate_images.flags.writeable
        for key in SEQUENCE_PARAMS:
            if key in payload['templates'][index]['params']:
                expected_values = tuple(payload['templates'][index]['params'][key]) + tuple(second['templates'][index]['params'][key])
                equal(expected_values, item['params'][key])
    path = tmp_path / 'combined.paintanalysis'
    save_analysis_session(path, actual)
    reopened = load_analysis_session(path, live._analysis_record_types())
    for a, b in zip(actual['templates'], reopened['templates']):
        for field in fields(a['picks']):
            equal(getattr(a['picks'], field.name), getattr(b['picks'], field.name))
        equal(a['params'], b['params'])


def test_append_releases_tile_arrays_and_preserves_nested_metadata(tmp_path):
    _, _, original = run_fixture()
    payload = deepcopy(original)
    nested = np.arange(400, dtype=float).reshape(200, 2)
    payload['templates'][0]['params']['nested_example'] = {'values': nested}
    payload['templates'][0]['params']['object_example'] = np.array([{'label': 'open'}, None], dtype=object)
    refs = [weakref.ref(nested)]
    for item in payload['templates']:
        refs.extend(weakref.ref(getattr(item['picks'], name)) for name in ('alignment_candidate_images', 'site_counts') if hasattr(item['picks'], name))
        refs.extend(weakref.ref(region) for region in item['picks'].regions)
    writer = StreamingTileResults(tmp_path, multi=True)
    writer.append(payload)
    del payload, nested, item
    gc.collect()
    assert all(ref() is None for ref in refs), 'Writer retained arrays from an earlier tile'
    restored = writer.finish()['templates'][0]
    np.testing.assert_equal(restored['params']['nested_example']['values'], np.arange(400).reshape(200, 2))
    np.testing.assert_equal(restored['params']['object_example'], np.array([{'label': 'open'}, None], dtype=object))
    assert not restored['picks'].regions[0].flags.writeable


def test_worker_disk_failure_preserves_checkpoints_and_can_rebuild(tmp_path, monkeypatch):
    from unittest.mock import Mock
    live, run, payload = run_fixture()
    live._identify_origami_worker = Mock(return_value=('origami_multi_picks', deepcopy(payload)))
    live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('*.paintanalysis')}
    with monkeypatch.context() as patch:
        patch.setattr(_Rows, 'append', Mock(side_effect=OSError(28, 'disk full')))
        with pytest.raises(OSError, match='disk full'):
            live._checkpointed_tiled_worker(tmp_path, run, completed_only=True)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('*.paintanalysis')}
    live._identify_origami_worker = Mock(side_effect=AssertionError('Recomputed saved tile'))
    _, result = live._checkpointed_tiled_worker(tmp_path, run, completed_only=True)
    np.testing.assert_equal(result['counts'], [24, 24])
    assert isinstance(result['points_nm'], np.memmap)


def test_disk_backed_results_render_all_classification_views(tmp_path, monkeypatch):
    from test_analysis_view_restore import matrix
    live, _, payload = run_fixture()
    writer = StreamingTileResults(tmp_path, multi=True)
    writer.append(payload)
    combined = writer.finish()
    live.origami_multi_template_results = {
        item['name']: {'picks': item['picks'], 'params': item['params']}
        for item in combined['templates']
    }
    live.origami_multi_template_overlay_results = {}
    # Exercise view builders against read-only, disk-backed arrays (including
    # density overlays, galleries and per-particle diagnostics for every class).
    snapshots = matrix(live, monkeypatch)
    assert snapshots


def test_small_array_writes_are_batched_per_tile(tmp_path, monkeypatch):
    from pathlib import Path
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    original_open = Path.open
    opens = []

    def track_open(path, *args, **kwargs):
        if path == store.path:
            opens.append(args)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track_open)
    values = [np.arange(i % 9, dtype=np.int32) for i in range(500)]
    values += [np.arange(24, dtype=float).reshape(4, 6).T, np.array(3.5), np.empty((3, 0))]
    with store.writing():
        saved = store.put(values, {})
    assert store.writer is None
    assert len(opens) == 1  # Hundreds of small arrays use one buffered stream.
    for expected, actual in zip(values, store.get(saved)):
        np.testing.assert_array_equal(actual, expected)


def test_array_writer_closes_after_failed_tile(tmp_path):
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    with pytest.raises(RuntimeError, match='interrupted'):
        with store.writing():
            stream = store.writer
            store.put(np.arange(4), {})
            raise RuntimeError('interrupted')
    assert stream.closed
    assert store.writer is None


@pytest.mark.parametrize('limit', [2, 8])
def test_result_writers_reuse_handles_and_bound_open_files(tmp_path, monkeypatch, limit):
    from pathlib import Path
    from tiled_aggregation import _Rows, _WriterPool
    pool = _WriterPool()
    pool.limit = limit
    original_open = Path.open
    opens = []

    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.suffix == '.bin':
            opens.append(handle)
        return handle

    monkeypatch.setattr(Path, 'open', track_open)
    rows = [_Rows(tmp_path / f'{i}.bin', pool) for i in range(3)]
    for tile in range(3):
        for row in rows:
            row.append(np.array([tile, tile + 10]))
            assert len(pool.handles) <= limit
    for row in rows:
        np.testing.assert_array_equal(row.finish(), [0, 10, 1, 11, 2, 12])
    assert len(opens) == (3 if limit == 8 else 9)
    assert all(handle.closed for handle in opens)
    assert not pool.handles
