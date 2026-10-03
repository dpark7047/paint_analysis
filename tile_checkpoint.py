"""Atomic per-tile recovery archives, tied to a self-contained run snapshot."""
from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile
import os

from analysis_session import load_analysis_session, save_analysis_session


@contextmanager
def checkpoint_lock(directory):
    """OS locks are released if the process crashes; no stale lock cleanup needed."""
    with (Path(directory) / 'run.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                handle.write(b'0')
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('This checkpoint folder is already in use by another analysis.') from exc
        yield


def load_checkpoint_run(directory, record_types):
    run = load_analysis_session(Path(directory) / 'run.paintanalysis', record_types)
    validate_checkpoint_run(run)
    return run


def validate_checkpoint_run(run):
    if not isinstance(run, dict) or run.get('checkpoint_version') != 1:
        raise ValueError('Not a supported tiled-analysis checkpoint folder.')
    if not all(key in run for key in ('run_id', 'analysis', 'context')):
        raise ValueError('The tiled-analysis checkpoint is incomplete.')
    if not isinstance(run['analysis'], dict) or not isinstance(run['context'], dict):
        raise ValueError('The tiled-analysis checkpoint has invalid source/settings data.')


class TileCheckpoint:
    def __init__(self, directory, run_id, record_types, report=lambda message: None, array_cache_dir=None):
        self.directory = Path(directory)
        self.run_id = run_id
        self.record_types = record_types
        self.report = report
        self.array_cache_dir = array_cache_dir

    def path(self, index):
        return self.directory / f'tile-{index:08d}.paintanalysis'

    def load(self, index, bounds, mode):
        path = self.path(index)
        if not path.exists():
            return None
        try:
            record = load_analysis_session(path, self.record_types, array_cache_dir=self.array_cache_dir)
            if (record['run_id'] != self.run_id or record['index'] != index
                    or tuple(record['bounds']) != tuple(bounds) or record['mode'] != mode):
                raise ValueError('Tile belongs to a different run or geometry.')
            return record['payload']
        except OSError as exc:
            if self.array_cache_dir is not None or exc.errno in {12, 13, 24, 28, 30}:
                raise OSError(f"Cannot create preview cache in {self.array_cache_dir or self.directory}: {exc}. Original checkpoints are unchanged.") from exc
            self.report(f'Skipping unreadable checkpoint for tile {index + 1}: {exc}')
            return None
        except (ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
            self.report(f'Skipping unreadable checkpoint for tile {index + 1}: {exc}')
            return None

    def save(self, index, bounds, mode, payload):
        try:
            save_analysis_session(self.path(index), dict(run_id=self.run_id, index=index,
                                  bounds=tuple(bounds), mode=mode, payload=payload))
        except OSError as exc:
            raise OSError(f'Could not checkpoint tile {index + 1} in {self.directory}. '
                          f'Previously saved tiles are retained; free space and resume. {exc}') from exc
        self.report(f'Checkpoint saved: tile {index + 1} — {self.directory}')
