"""Versioned, data-only analysis archives. No executable pickle payloads."""
from dataclasses import fields, is_dataclass
from pathlib import Path
import json
import os
import tempfile
import hashlib
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

FORMAT = 'paint-analysis-session'
VERSION = 1


def save_analysis_session(path, state):
    """Write atomically, retaining shared arrays only once across templates."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            nodes, memo, retained = [], {}, []

            def encode(value):
                if isinstance(value, np.generic):
                    value = value.item()
                if value is None or isinstance(value, (bool, str, int, float)):
                    return value
                if id(value) in memo:
                    return {'ref': memo[id(value)]}
                index = len(nodes)
                memo[id(value)] = index
                retained.append(value)
                nodes.append(None)
                if isinstance(value, np.ndarray):
                    if value.dtype.hasobject:
                        node = dict(kind='object_array', shape=value.shape, values=[encode(v) for v in value.flat])
                    else:
                        name = f'arrays/{index}.npy'
                        with archive.open(name, 'w', force_zip64=True) as stream:
                            np.save(stream, value, allow_pickle=False)
                        node = dict(kind='array', file=name)
                elif isinstance(value, Path):
                    node = dict(kind='path', value=str(value))
                elif isinstance(value, pd.DataFrame):
                    node = dict(kind='dataframe', columns=encode(list(value.columns)),
                                index=encode(value.index.to_numpy()), index_name=encode(value.index.name),
                                values=[encode(value.iloc[:, i].to_numpy()) for i in range(len(value.columns))],
                                dtypes=[str(dtype) for dtype in value.dtypes])
                elif is_dataclass(value) and not isinstance(value, type):
                    node = dict(kind='record', name=type(value).__name__,
                                values={field.name: encode(getattr(value, field.name)) for field in fields(value)})
                elif isinstance(value, dict):
                    node = dict(kind='dict', values=[[encode(k), encode(v)] for k, v in value.items()])
                elif isinstance(value, (list, tuple, set)):
                    node = dict(kind=type(value).__name__, values=[encode(v) for v in value])
                else:
                    raise TypeError(f'Cannot save analysis value of type {type(value).__name__}')
                nodes[index] = node
                return {'ref': index}

            root = encode(state)
            archive.writestr('manifest.json', json.dumps(dict(format=FORMAT, version=VERSION, root=root, nodes=nodes)))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_analysis_session(path, record_types, progress_callback=None, array_cache_dir=None):
    """Decode only known data records; reject unsupported archive versions."""
    cache_directory = None
    if array_cache_dir is not None:
        path = Path(path)
        info = path.stat()
        identity = hashlib.sha256(f"{path.resolve()}:{info.st_size}:{info.st_mtime_ns}".encode()).hexdigest()[:24]
        cache_directory = Path(array_cache_dir) / identity
        cache_directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive:
        total_bytes = max(1, sum(entry.file_size for entry in archive.infolist()))
        bytes_read, nodes_done, last_percent = 0, 0, -1
        node_count = 1

        def report(finished=False):
            nonlocal last_percent
            percent = 100 if finished else min(99, int(90 * bytes_read / total_bytes + 9 * nodes_done / node_count))
            if progress_callback is not None and percent != last_percent:
                last_percent = percent
                progress_callback(percent)

        class ProgressReader:
            def __init__(self, stream):
                self.stream = stream
                self.high_water = 0

            def read(self, size=-1):
                nonlocal bytes_read
                # Bound reads to report progress even for a single large array.
                chunks = []
                remaining = size
                while remaining != 0:
                    data = self.stream.read(min(remaining, 1024 * 1024) if remaining > 0 else 1024 * 1024)
                    if not data:
                        break
                    chunks.append(data)
                    position = self.stream.tell()
                    bytes_read += max(0, position - self.high_water)
                    self.high_water = max(position, self.high_water)
                    report()
                    if remaining > 0:
                        remaining -= len(data)
                return b''.join(chunks)

            def seek(self, *args):
                return self.stream.seek(*args)

            def tell(self):
                return self.stream.tell()

        report()
        with archive.open('manifest.json') as stream:
            manifest = json.loads(ProgressReader(stream).read())
        if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
            raise ValueError('Unsupported analysis file format or version.')
        nodes, cache, decoding = manifest['nodes'], {}, set()
        node_count = max(1, len(nodes))

        def decode(value):
            nonlocal nodes_done
            if not isinstance(value, dict):
                return value
            index = value['ref']
            if index in cache:
                return cache[index]
            if index in decoding:
                raise ValueError('Cyclic analysis records are not supported.')
            decoding.add(index)
            node = nodes[index]
            kind = node['kind']
            if kind == 'array':
                member = archive.getinfo(node['file'])
                if cache_directory is not None and member.file_size >= 64 * 1024:
                    cached_path = cache_directory / f"array-{index}.npy"
                    if not cached_path.exists() or cached_path.stat().st_size != member.file_size:
                        temporary = None
                        try:
                            with tempfile.NamedTemporaryFile(dir=cache_directory, delete=False) as target:
                                temporary = Path(target.name)
                                with archive.open(member) as stream:
                                    shutil.copyfileobj(ProgressReader(stream), target, length=1024 * 1024)
                            os.replace(temporary, cached_path)
                        finally:
                            if temporary is not None:
                                temporary.unlink(missing_ok=True)
                    result = np.load(cached_path, mmap_mode='r', allow_pickle=False)
                else:
                    with archive.open(member) as stream:
                        result = np.load(ProgressReader(stream), allow_pickle=False)
            elif kind == 'object_array':
                result = np.empty(node['shape'], dtype=object)
                if result.size != len(node['values']):
                    raise ValueError('Invalid object array size.')
                for i, item in enumerate(node['values']):
                    result.flat[i] = decode(item)
            elif kind == 'path':
                result = Path(node['value'])
            elif kind == 'dataframe':
                columns = decode(node['columns'])
                result = pd.DataFrame({i: pd.Series(decode(v)).astype(dtype)
                                       for i, (v, dtype) in enumerate(zip(node['values'], node['dtypes']))})
                result.columns = columns
                result.index = pd.Index(decode(node['index']), name=decode(node['index_name']))
            elif kind == 'record':
                if node['name'] not in record_types:
                    raise ValueError(f"Unsupported analysis record: {node['name']}")
                result = record_types[node['name']](**{k: decode(v) for k, v in node['values'].items()})
            elif kind == 'dict':
                result = {decode(k): decode(v) for k, v in node['values']}
            elif kind in ('list', 'tuple', 'set'):
                result = {'list': list, 'tuple': tuple, 'set': set}[kind](decode(v) for v in node['values'])
            else:
                raise ValueError(f'Unsupported analysis node: {kind}')
            cache[index] = result
            decoding.remove(index)
            nodes_done += 1
            report()
            return result

        result = decode(manifest['root'])
        report(finished=True)
        return result
