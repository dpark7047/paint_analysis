"""Incremental tiled results: keep metadata in memory and numeric data on disk."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
import tempfile

import numpy as np

from origami_analysis import OrigamiPickResult


SEQUENCE_PARAMS = (
    'classification_observed_digital_states', 'classification_exact_digital_match',
    'classification_lattice_precision', 'classification_empty_cell_fraction',
    'classification_bright_cell_probability', 'classification_cell_pattern_correlation',
    'classification_cell_agreement', 'classification_monte_carlo_posterior',
    'classification_monte_carlo_log_bayes_factor', 'classification_scores',
    'classification_dispositions', 'classification_qc_eligible',
    'classification_lookup_eligible', '_alignment_accepted_mask',
    'classification_winner_margins', 'classification_runner_up_templates',
    'classification_template_probabilities', 'classification_raw_template_probabilities',
    'classification_template_probability_vectors', 'classification_raw_template_probability_vectors',
    'classification_crop_retained_fraction', 'classification_logical_bit_probabilities',
    'digital_group_localization_evidence', 'digital_group_prominences', 'digital_pixel_probabilities',
)


@dataclass
class _ArrayRef:
    offset: int
    shape: tuple
    dtype: np.dtype


@dataclass
class _ObjectArrayRef:
    shape: tuple
    values: list


class _ArrayStore:
    """One shared file for variable-size region, history, and parameter arrays."""
    def __init__(self, directory):
        self.path = directory / 'arrays.bin'
        self.path.touch()
        self.size = 0
        self.buffer = None
        self.writer = None

    @contextmanager
    def writing(self):
        # One buffered stream per tile. ndarray.tofile bypasses Python buffering
        # and performs seek/flush work for each tiny array on external drives.
        with self.path.open('ab', buffering=1024 * 1024) as handle:
            self.writer = handle
            try:
                yield
            finally:
                self.writer = None

    def put(self, value, memo):
        if isinstance(value, np.ndarray):
            key = id(value)
            if key in memo:
                return memo[key]
            if value.dtype.hasobject:
                # Object values are metadata, never raw Python pointers on disk.
                result = _ObjectArrayRef(value.shape, [self.put(item, memo) for item in value.flat])
                memo[key] = result
                return result
            offset = (self.size + 63) // 64 * 64
            if self.writer is None:
                raise RuntimeError('Array storage requires an active tile write.')
            self.writer.write(b'\0' * (offset - self.size))
            if value.nbytes:
                self.writer.write(memoryview(np.ascontiguousarray(value)).cast('B'))
            self.size = offset + value.nbytes
            result = _ArrayRef(offset, value.shape, value.dtype)
            memo[key] = result
            return result
        if isinstance(value, dict):
            return {key: self.put(item, memo) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.put(item, memo) for item in value)
        return value

    def get(self, value):
        if isinstance(value, _ObjectArrayRef):
            output = np.empty(value.shape, dtype=object)
            for index, item in enumerate(value.values):
                output.flat[index] = self.get(item)
            return output
        if isinstance(value, _ArrayRef):
            if not np.prod(value.shape, dtype=np.int64):
                return np.empty(value.shape, dtype=value.dtype)
            if self.buffer is None:
                self.buffer = np.memmap(self.path, mode='r', dtype=np.uint8)
            return np.ndarray(value.shape, dtype=value.dtype, buffer=self.buffer, offset=value.offset)
        if isinstance(value, dict):
            return {key: self.get(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.get(item) for item in value)
        return value


class _WriterPool:
    """Bound buffered writers and descriptors while reusing files across tiles."""
    def __init__(self):
        self.limit = 512
        try:
            import resource
            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft > 0:
                self.limit = min(self.limit, max(16, (soft - 64) // 2))
        except (ImportError, ValueError):
            self.limit = 64
        self.handles = OrderedDict()

    def get(self, path):
        handle = self.handles.pop(path, None)
        if handle is None:
            if len(self.handles) >= self.limit:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
            handle = path.open('ab', buffering=64 * 1024)
        self.handles[path] = handle
        return handle

    def close_file(self, path):
        handle = self.handles.pop(path, None)
        if handle is not None:
            handle.close()

    def close(self):
        error = None
        while self.handles:
            _, handle = self.handles.popitem()
            try:
                handle.close()
            except OSError as exc:
                error = error or exc
        if error is not None:
            raise error


class _Rows:
    """Append rows without allocating the final array or keeping prior tiles."""
    def __init__(self, path, pool=None):
        self.path = path
        self.pool = pool
        self.shape = None
        self.dtype = None
        self.count = 0
        self.empty = None

    def append(self, value):
        value = np.asarray(value)
        if self.empty is None:
            self.empty = np.empty((0, *value.shape[1:]), dtype=value.dtype)
        if not len(value):
            return
        if value.dtype.hasobject:
            raise ValueError('A numeric tiled result unexpectedly contains object values.')
        if self.shape is None:
            self.shape, self.dtype = value.shape[1:], value.dtype
        if value.shape[1:] != self.shape or value.dtype != self.dtype:
            raise ValueError('Tile array shape or dtype changed during disk aggregation.')
        if value.nbytes:
            data = memoryview(np.ascontiguousarray(value)).cast('B')
            if self.pool is not None:
                self.pool.get(self.path).write(data)
            else:
                with self.path.open('ab') as handle:
                    handle.write(data)
        self.count += len(value)

    def finish(self):
        if self.pool is not None:
            self.pool.close_file(self.path)
        if not self.count:
            return self.empty
        shape = (self.count, *self.shape)
        if not np.prod(shape, dtype=np.int64):
            return np.empty(shape, dtype=self.dtype)
        return np.memmap(self.path, mode='r', dtype=self.dtype, shape=shape)


class _Picks:
    _lists = {'regions', 'aligned_regions', 'alignment_pose_history'}
    _density = {'density_image', 'density_contrast', 'density_component_labels'}
    _static_arrays = {'template_points_nm', 'alignment_reference_image'}

    def __init__(self, directory, store, pool):
        self.directory, self.store, self.pool = directory, store, pool
        self.values, self.rows = {}, {}
        self.extent = None
        self.grid = None

    def append(self, picks, memo):
        grid = picks.template_points_nm
        if self.grid is None:
            self.grid = np.array(grid, copy=True)
        elif grid.shape != self.grid.shape or not np.allclose(grid, self.grid, rtol=0, atol=1e-9):
            raise ValueError('Tiled origami results must use the same theoretical template.')
        extent = picks.density_extent_nm
        if self.extent is None:
            self.extent = tuple(extent)
        else:
            old = self.extent
            self.extent = (min(old[0], extent[0]), max(old[1], extent[1]),
                           min(old[2], extent[2]), max(old[3], extent[3]))
        for field in fields(OrigamiPickResult):
            name = field.name
            value = getattr(picks, name)
            if name == 'density_extent_nm':
                continue
            if name in self._lists:
                if name == 'alignment_pose_history' and not value:
                    value = [[] for _ in picks.regions]
                self.values.setdefault(name, []).extend(self.store.put(value, memo))
            elif name in self._density:
                self.values[name] = np.empty((0, 0), dtype=np.asarray(value).dtype)
            elif isinstance(value, np.ndarray) and name not in self._static_arrays:
                if name == 'footprint_overlap_fraction' and len(value) != len(picks.regions):
                    value = np.zeros(len(picks.regions))
                if name not in self.rows:
                    self.rows[name] = _Rows(self.directory / f'{name}.bin', self.pool)
                self.rows[name].append(value)
            elif name not in self.values:
                self.values[name] = self.store.put(value, memo)

    def finish(self):
        values = self.store.get(self.values)
        values.update({name: rows.finish() for name, rows in self.rows.items()})
        values['density_extent_nm'] = self.extent
        return OrigamiPickResult(**values)


class StreamingTileResults:
    """Append one tile, discard its payload, then expose combined disk-backed arrays.

    Cache files are disposable. The atomic per-tile archives remain the recovery
    authority, so interrupted builders are never mistaken for complete results.
    """
    def __init__(self, parent, multi):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='combined-', dir=parent))
        self.store = _ArrayStore(self.directory)
        self.pool = _WriterPool()
        self.multi = multi
        self.templates = []
        self.names = None
        self.counts = None
        self.unclassified = self.suppressed = 0
        self.details = []
        self.centers = _Rows(self.directory / 'unclassified-centers.bin', self.pool)
        self.tiles = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.pool.close()
        except OSError:
            if exc_type is None:
                raise

    def append(self, payload):
        with self.store.writing():
            self._append(payload)

    def _append(self, payload):
        items = payload['templates'] if self.multi else [dict(name='single', picks=payload['picks'], params={})]
        names = [item['name'] for item in items]
        if self.names is None:
            self.names = names
            for index, item in enumerate(items):
                directory = self.directory / str(index)
                directory.mkdir()
                self.templates.append(dict(picks=_Picks(directory, self.store, self.pool), params=None, sequences={}))
        if names != self.names:
            raise ValueError('Template order changed while aggregating tiled classifications.')
        # IDs are only meaningful within this tile; never retain its arrays.
        memo = {}
        for item, target in zip(items, self.templates):
            target['picks'].append(item['picks'], memo)
            if target['params'] is None:
                target['params'] = self.store.put({k: v for k, v in item['params'].items() if k not in SEQUENCE_PARAMS}, memo)
            for name in SEQUENCE_PARAMS:
                if name in item['params']:
                    target['sequences'].setdefault(name, []).extend(self.store.put(tuple(item['params'][name]), memo))
        if self.multi:
            counts = np.asarray(payload['counts'], dtype=int)
            self.counts = counts.copy() if self.counts is None else self.counts + counts
            self.unclassified += int(payload['unclassified_count'])
            self.suppressed += int(payload.get('suppressed_duplicate_count', 0))
            self.centers.append(np.asarray(payload.get('unclassified_centers_nm', np.empty((0, 2))), dtype=float).reshape(-1, 2))
            self.details.extend(self.store.put(payload.get('unclassified_details', []), memo))
        self.tiles += 1

    def finish(self):
        if not self.tiles:
            raise ValueError('No completed tiles to combine.')
        templates = []
        for name, target in zip(self.names, self.templates):
            params = self.store.get(target['params'])
            params.update({key: tuple(self.store.get(value)) for key, value in target['sequences'].items()})
            templates.append(dict(name=name, picks=target['picks'].finish(), params=params))
        if not self.multi:
            return dict(picks=templates[0]['picks'])
        return dict(templates=templates, counts=self.counts, unclassified_count=self.unclassified,
                    suppressed_duplicate_count=self.suppressed, unclassified_centers_nm=self.centers.finish(),
                    unclassified_details=self.store.get(self.details))

    def source_points(self, locs, pixelsize):
        rows = _Rows(self.directory / 'source-points.bin', self.pool)
        for start in range(0, len(locs), 100_000):
            rows.append(locs.iloc[start:start + 100_000][['x', 'y']].to_numpy(dtype=float) * pixelsize)
        return rows.finish()
