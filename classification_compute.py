"""Optional numerical acceleration for origami analysis only.

The reference algorithms, thresholds, coordinates and Picasso drift routines
remain in their original modules. CPU is the default; the experimental launcher
selects auto. All public results are NumPy arrays, including on the GPU path.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import time
import traceback
import sys
import warnings

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree

_mode = ContextVar("classification_compute", default=None)
_gpu = None
_failure = None
_lock = threading.RLock()
_counts = {}


def mode():
    value = _mode.get() or os.environ.get("PAINT_CLASSIFICATION_COMPUTE", "cpu")
    if value not in {"reference", "cpu", "auto", "gpu"}:
        raise ValueError("PAINT_CLASSIFICATION_COMPUTE must be reference, cpu, auto or gpu")
    return value


@contextmanager
def use_mode(value):
    token = _mode.set(value)
    try:
        mode()
        yield
    finally:
        _mode.reset(token)


def cpu_only(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Preserve the reference switch used by the comparison harness.
        with use_mode("reference" if mode() == "reference" else "cpu"):
            return function(*args, **kwargs)
    return wrapped


def gpu_module():
    global _gpu, _failure
    if mode() in {"reference", "cpu"}:
        return None
    with _lock:
        if _failure is not None:
            if mode() == "gpu":
                raise RuntimeError(f"Classification GPU unavailable: {_failure}")
            return None
        if _gpu is None:
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("No CUDA device found")
                # Test compilation as well as driver/device discovery.
                cp.asnumpy(cp.arange(4, dtype=cp.float64) ** 2)
                _gpu = cp
            except Exception as exc:
                _failure = str(exc)
                if mode() == "gpu":
                    raise RuntimeError(f"Classification GPU unavailable: {_failure}") from exc
                warnings.warn(f"Classification uses CPU: {_failure}", RuntimeWarning)
                return None
        return _gpu


def _gpu_failed(exc):
    global _failure
    if mode() == "gpu":
        raise exc
    with _lock:
        first = _failure is None
        _failure = str(exc)
    if first:
        warnings.warn(f"Classification GPU failed; retrying on CPU: {exc}", RuntimeWarning)


def _count(operation, backend):
    with _lock:
        key = f"{operation}:{backend}"
        _counts[key] = _counts.get(key, 0) + 1


def status():
    with _lock:
        return {"mode": mode(), "gpu_initialized": _gpu is not None,
                "fallback_reason": _failure, "operations": dict(_counts)}


def gaussian_filter(image, *args, **kwargs):
    # Small thumbnails are faster on the CPU. Large detection/overlay maps
    # benefit from GPU filtering; batch site images use a separate resident path.
    cp = gpu_module() if np.size(image) >= 262144 else None
    if cp is not None:
        try:
            from cupyx.scipy.ndimage import gaussian_filter as gpu_filter
            result = cp.asnumpy(gpu_filter(cp.asarray(image), *args, **kwargs))
            _count("gaussian_filter", "gpu")
            return result
        except Exception as exc:
            _gpu_failed(exc)
    _count("gaussian_filter", "cpu")
    return ndimage.gaussian_filter(image, *args, **kwargs)


def fftconvolve(image, kernel, *args, **kwargs):
    cp = gpu_module() if np.size(image) >= 262144 else None
    if cp is not None:
        try:
            from cupyx.scipy.signal import fftconvolve as gpu_convolve
            result = cp.asnumpy(gpu_convolve(cp.asarray(image), cp.asarray(kernel), *args, **kwargs))
            _count("fftconvolve", "gpu")
            return result
        except Exception as exc:
            _gpu_failed(exc)
    _count("fftconvolve", "cpu")
    return signal.fftconvolve(image, kernel, *args, **kwargs)


@lru_cache(maxsize=64)
def _cached_template_tree(data, shape):
    # Copy into a read-only owned array, so callers cannot mutate cached geometry.
    points = np.frombuffer(data, dtype=np.float64).reshape(shape).copy()
    points.setflags(write=False)
    return cKDTree(points)


def template_tree(points):
    """Reuse the exact same SciPy tree for fixed template coordinates."""
    if mode() == "reference":
        return cKDTree(points)
    points = np.asarray(points, dtype=np.float64)
    return _cached_template_tree(points.tobytes(), points.shape)


def nearest_assignments(points, grid, *, use_gpu=True):
    """Original brute-force site assignment, in bounded point chunks.

    NumPy argmin's first-site tie rule is preserved. Radius/tie boundary values
    are recomputed on CPU so a CUDA rounding difference cannot alter support.
    """
    points, grid = np.asarray(points, dtype=float), np.asarray(grid, dtype=float)
    if not len(points):
        return np.empty(0), np.empty(0, dtype=int)
    cp = gpu_module() if use_gpu else None
    xp = np if cp is None else cp
    distances = np.empty(len(points))
    indices = np.empty(len(points), dtype=int)
    # Bound the broadcast difference tensor to roughly 32 MiB.
    chunk = max(1, min(65536, 2_000_000 // max(1, len(grid))))
    try:
        device_grid = xp.asarray(grid)
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            diff = xp.asarray(points[start:stop])[:, None, :] - device_grid[None, :, :]
            matrix = xp.sqrt(xp.sum(diff * diff, axis=2))
            nearest = xp.argmin(matrix, axis=1)
            values = matrix[xp.arange(stop - start), nearest]
            if cp is not None:
                nearest, values = cp.asnumpy(nearest), cp.asnumpy(values)
                # Exact ties and nearly equal choices use the original CPU rule.
                near_ties = cp.asnumpy(cp.sum(cp.abs(matrix - cp.asarray(values)[:, None]) <= 1e-12, axis=1)) > 1
                if np.any(near_ties):
                    cpu_matrix = np.linalg.norm(points[start:stop][near_ties, None, :] - grid[None, :, :], axis=2)
                    nearest[near_ties] = np.argmin(cpu_matrix, axis=1)
                    values[near_ties] = cpu_matrix[np.arange(len(cpu_matrix)), nearest[near_ties]]
            distances[start:stop], indices[start:stop] = values, nearest
        _count("nearest_assignments", "cpu" if cp is None else "gpu")
        return distances, indices
    except Exception as exc:
        if cp is None:
            raise
        _gpu_failed(exc)
        return nearest_assignments(points, grid, use_gpu=False)


def group_evidence(regions, grid, groups, radius, factors):
    """Batch direct group affinities; overlapping groups remain independent."""
    cp = gpu_module()
    if cp is None:
        return None
    try:
        result = np.zeros((len(regions), len(groups)))
        device_grid = cp.asarray(grid, dtype=cp.float64)
        group_sizes = cp.asarray([len(cells) for cells in groups], dtype=cp.float64)
        device_factors = cp.asarray(factors)
        # No candidate's full distance matrix is retained across point chunks.
        for start in range(0, len(regions), 32):
            batch = [np.asarray(region, dtype=float) for region in regions[start:start + 32]]
            batch = [region if region.ndim == 2 and region.shape[1:] == (2,)
                     else np.empty((0, 2)) for region in batch]
            sizes = np.asarray([len(region) for region in batch], dtype=int)
            if not np.sum(sizes):
                continue
            points = np.concatenate(batch)
            boundaries = np.cumsum(np.r_[0, sizes])
            evidence = cp.zeros((len(batch), len(groups)), dtype=cp.float64)
            chunk = max(1, min(65536, 2_000_000 // max(1, len(grid))))
            for point_start in range(0, len(points), chunk):
                point_stop = min(point_start + chunk, len(points))
                diff = cp.asarray(points[point_start:point_stop])[:, None, :] - device_grid[None, :, :]
                d2 = cp.sum(diff * diff, axis=2)
                weights = cp.stack([cp.min(d2[:, cells], axis=1) for cells in groups], axis=1)
                # Compare sqrt(distance) to radius, as in the reference path.
                inside = cp.sqrt(weights) <= radius
                weights = cp.where(inside, cp.exp(-0.5 * weights / (radius * radius)), 0.0)
                # Recheck radius boundaries with cKDTree (same semantics as reference).
                close = cp.asnumpy(cp.any(cp.abs(cp.sqrt(cp.stack([cp.min(d2[:, cells], axis=1) for cells in groups], axis=1)) - radius) <= 1e-12, axis=1))
                if np.any(close):
                    from scipy.spatial import cKDTree
                    host_points = points[point_start:point_stop][close]
                    host_weights = np.zeros((len(host_points), len(groups)))
                    for group_index, cells in enumerate(groups):
                        distance = cKDTree(grid[cells]).query(host_points, k=1)[0]
                        keep = distance <= radius
                        host_weights[keep, group_index] = np.exp(-0.5 * distance[keep] ** 2 / radius ** 2)
                    weights[cp.asarray(close)] = cp.asarray(host_weights)
                for index in range(len(batch)):
                    lo, hi = max(boundaries[index], point_start), min(boundaries[index + 1], point_stop)
                    if hi > lo:
                        evidence[index] += cp.sum(weights[lo - point_start:hi - point_start], axis=0)
            result[start:start + len(batch)] = cp.asnumpy(evidence / group_sizes / device_factors)
        _count("direct_group_evidence", "gpu")
        return result
    except Exception as exc:
        _gpu_failed(exc)
        return None


def peak_diagnostics(density, x_centers, y_centers, grid, counts, nearest_spacing,
                     radius, bandwidth, x_min, y_min, effective_x, effective_y):
    """Vectorize the existing independent peak/boundary measurements over sites.

    Density creation remains unchanged. A single interpolation and percentile
    operation replaces the per-site calls, including NumPy's linear percentile.
    """
    sample = ndimage.map_coordinates
    # These site patches are small: vectorized CPU is usually faster than a
    # separate GPU transfer for every candidate.
    sites = len(grid)
    peak_positions = np.full((sites, 2), np.nan)
    peak_densities = np.zeros(sites)
    boundary_densities = np.zeros((sites, 32))
    reference_densities = np.zeros(sites)
    reference_positions = np.full((sites, 2), np.nan)
    boundary_points = np.full((sites, 32, 2), np.nan)
    prominence = np.zeros(sites)
    search_radius = np.minimum(radius, .40 * nearest_spacing)
    x_masks = np.abs(x_centers[None, :] - grid[:, 0, None]) <= search_radius[:, None]
    y_masks = np.abs(y_centers[None, :] - grid[:, 1, None]) <= search_radius[:, None]
    selected = np.flatnonzero((counts > 0) & x_masks.any(axis=1) & y_masks.any(axis=1))
    if not len(selected):
        return prominence, peak_positions, peak_densities, boundary_densities, reference_densities, reference_positions, boundary_points
    x_masks, y_masks = x_masks[selected], y_masks[selected]
    x_start, y_start = x_masks.argmax(axis=1), y_masks.argmax(axis=1)
    x_length, y_length = x_masks.sum(axis=1), y_masks.sum(axis=1)
    xx = x_start[:, None] + np.arange(x_length.max())[None, :]
    yy = y_start[:, None] + np.arange(y_length.max())[None, :]
    xx = np.minimum(xx, len(x_centers) - 1)
    yy = np.minimum(yy, len(y_centers) - 1)
    patches = density[yy[:, :, None], xx[:, None, :]].copy()
    inside = ((x_centers[xx][:, None, :] - grid[selected, 0, None, None]) ** 2
              + (y_centers[yy][:, :, None] - grid[selected, 1, None, None]) ** 2
              <= search_radius[selected, None, None] ** 2)
    inside &= ((np.arange(xx.shape[1])[None, None, :] < x_length[:, None, None])
               & (np.arange(yy.shape[1])[None, :, None] < y_length[:, None, None]))
    patches[~inside] = -np.inf
    flat = patches.reshape(len(selected), -1).argmax(axis=1)
    row, col = np.unravel_index(flat, patches.shape[1:])
    positions = np.column_stack((x_centers[xx[np.arange(len(selected)), col]],
                                 y_centers[yy[np.arange(len(selected)), row]]))
    peaks = patches[np.arange(len(selected)), row, col]
    boundary_radius = np.minimum(.48 * nearest_spacing[selected], max(radius, 2.5 * bandwidth))
    angles = 2.0 * np.pi * np.arange(32, dtype=float) / 32.0
    boundary = positions[:, None, :] + boundary_radius[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))[None, :, :]
    sampled = sample(density, ((boundary[:, :, 1] - y_min) / effective_y - .5,
                               (boundary[:, :, 0] - x_min) / effective_x - .5),
                     order=1, mode="constant", cval=0.0)
    saddle = np.percentile(sampled, 90.0, axis=1)
    references = np.abs(sampled - saddle[:, None]).argmin(axis=1)
    peak_positions[selected], peak_densities[selected] = positions, peaks
    boundary_densities[selected], reference_densities[selected] = sampled, saddle
    reference_positions[selected] = boundary[np.arange(len(selected)), references]
    boundary_points[selected] = boundary
    valid = peaks > 1e-12
    prominence[selected[valid]] = np.clip((peaks[valid] - saddle[valid]) / peaks[valid], 0., 1.)
    _count("site_peak_diagnostics", "cpu-vectorized")
    return prominence, peak_positions, peak_densities, boundary_densities, reference_densities, reference_positions, boundary_points


def _log(record):
    destination = os.environ.get("PAINT_CLASSIFICATION_LOG")
    if destination:
        try:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _lock, path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Diagnostic logging must never interrupt an analysis.


def timed(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        _log({"event": "start", "operation": function.__name__})
        try:
            return function(*args, **kwargs)
        finally:
            _log({"event": "finish", "operation": function.__name__,
                  "seconds": time.perf_counter() - start, "compute": status()})
    return wrapped


def worker_watchdog(stop, operation):
    # Records both UI and worker stacks when a job takes long enough to inspect.
    # Runs outside Tk and does not touch any GUI object.
    while not stop.wait(30):
        _log({"event": "worker_running", "operation": operation,
              "stacks": {str(identifier): "".join(traceback.format_stack(frame))
                         for identifier, frame in sys._current_frames().items()},
              "compute": status()})
