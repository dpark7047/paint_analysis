"""Versioned, data-only analysis archives. No executable pickle payloads."""
from dataclasses import fields, is_dataclass
from pathlib import Path
import json
import os
import tempfile
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


def load_analysis_session(path, record_types):
    """Decode only known data records; reject unsupported archive versions."""
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
            raise ValueError('Unsupported analysis file format or version.')
        nodes, cache, decoding = manifest['nodes'], {}, set()

        def decode(value):
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
                with archive.open(node['file']) as stream:
                    result = np.load(stream, allow_pickle=False)
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
            return result

        return decode(manifest['root'])
