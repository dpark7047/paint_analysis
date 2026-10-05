"""Synthetic-only benchmark and CPU/GPU result checks. Never opens user data."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import classification_compute as compute
from origami_analysis import (ideal_grid_points, iter_sparse_site_evidence,
                             _align_regions_by_image_correlation, pick_origami_candidates,
                             align_picked_origamis, classify_template_candidates)
from paint_analysis_gui import PaintAnalysisApp, exact_digital_template_matches


def measure(function, selected_mode, repeats):
    with compute.use_mode(selected_mode):
        before = dict(compute.status()["operations"])
        function()  # warm up CUDA compilation, FFT plans and memory pools
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = function()
            seconds.append(time.perf_counter() - start)
        after = compute.status()["operations"]
        operations = {key: value - before.get(key, 0) for key, value in after.items()
                      if value != before.get(key, 0)}
    return result, {"seconds": seconds, "median_seconds": float(np.median(seconds)),
                    "operations": operations}


def compare_sites(expected, actual):
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        for field in fields(left):
            np.testing.assert_allclose(getattr(left, field.name), getattr(right, field.name),
                                       rtol=1e-12, atol=1e-12, err_msg=field.name)


def compare_detection(expected, actual):
    assert len(expected[0]) == len(actual[0]), "Candidate count changed"
    # FFT noise can reorder equal-score independent candidates. Compare each
    # candidate's complete source coordinates, independently of that ranking.
    def canonical(regions):
        return sorted([region[np.lexsort((region[:, 1], region[:, 0]))] for region in regions],
                      key=lambda region: tuple(np.mean(region, axis=0)))
    for left, right in zip(canonical(expected[0]), canonical(actual[0])):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(expected[1], actual[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(expected[2], actual[2], rtol=1e-10, atol=1e-12)


def compare_alignment(expected, actual):
    for left, right in zip(expected[0], actual[0]):
        np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-5)
    for index in range(1, 5):
        np.testing.assert_allclose(expected[index], actual[index], rtol=1e-7, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(".gpu-results/benchmark.json"))
    parser.add_argument("--modes", nargs="+", choices=("reference", "cpu", "gpu", "auto"),
                        default=["reference", "cpu", "gpu", "auto"],
                        help="Use --modes reference cpu auto to benchmark without requiring CUDA.")
    args = parser.parse_args()
    if args.modes[0] != "reference" or len(set(args.modes)) != len(args.modes):
        parser.error("modes must start with reference and contain no duplicates")
    if args.repeats < 1 or args.candidates < 1:
        parser.error("repeats and candidates must be positive")
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path(".gpu-state/cupy").resolve()))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".gpu-state/numba").resolve()))
    properties = None
    if "gpu" in args.modes:
        with compute.use_mode("gpu"):
            cp = compute.gpu_module()  # Forced GPU benchmarks must never silently use CPU.
            properties = cp.cuda.runtime.getDeviceProperties(0)
    rng = np.random.default_rng(42)
    grid = ideal_grid_points(8, 12, 10.909, 5.714)
    regions = [np.repeat(grid, 30, axis=0) + rng.normal(0, 1, (len(grid) * 30, 2))
               for _ in range(args.candidates)]
    # Large FOV but a modest number of fiducial particles for Step 1.
    marks = np.array([[-45., -12.], [45., -12.], [-45., 12.], [45., 12.]])
    detection_points = np.concatenate([
        np.repeat(marks, 12, axis=0) + rng.normal(0, 1, (48, 2)) + [x * 700, y * 700]
        for x in range(9) for y in range(9)])
    small_grid = ideal_grid_points(3, 4, 15., 12.)
    alignment_regions = []
    for index in range(12):
        angle = np.deg2rad(13 + index * 30)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = np.repeat(small_grid, 8, axis=0) + rng.normal(0, .7, (len(small_grid) * 8, 2))
        alignment_regions.append(points @ rotation.T + [index * 200, 100])
    groups = [tuple(range(index, index + 12)) for index in range(0, 96, 12)]
    model = dict(bit_ids=tuple(f"row{index}" for index in range(8)),
                 bit_physical_cells=groups, bit_cells=groups,
                 bit_brightness_factors=(1.,) * 8, physical_shape=(8, 12))
    classification_regions = []
    for index in range(args.candidates):
        pattern = np.array([(index >> bit) & 1 for bit in range(8)], dtype=bool)
        cells = np.concatenate([groups[bit] for bit in np.flatnonzero(pattern)]) if np.any(pattern) else np.empty(0, int)
        points = np.repeat(grid[cells], 30, axis=0)
        classification_regions.append(points + rng.normal(0, 1, points.shape))

    def classify():
        params = dict(rows=8, columns=12, spacing_x_nm=10.909, spacing_y_nm=5.714,
                      site_mask_radius_nm=2.5, min_site_localizations=3,
                      min_site_evidence=.1, digital_pixel_model=model)
        PaintAnalysisApp._measure_direct_digital_groups(SimpleNamespace(aligned_regions=classification_regions), params)
        masks = []
        for index in range(128):
            params["logical_model"] = dict(model, active_bits=tuple(bool((index >> bit) & 1) for bit in range(8)))
            masks.append(exact_digital_template_matches(params, args.candidates))
        centers = np.column_stack((np.arange(args.candidates) * 200., np.zeros(args.candidates)))
        outcome = classify_template_candidates([centers] * len(masks), masks,
                    [np.where(mask, 0., -np.inf) for mask in masks], match_distance_nm=1., require_unique_match=True)
        return np.asarray(params["digital_group_localization_evidence"]), outcome

    def compare_classification(expected, actual):
        np.testing.assert_allclose(expected[0], actual[0], rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(expected[1].winning_template_indices, actual[1].winning_template_indices)
        np.testing.assert_array_equal(expected[1].counts, actual[1].counts)
        assert expected[1].unclassified_count == actual[1].unclassified_count

    def compare_overlay(expected, actual):
        np.testing.assert_array_equal(expected.site_counts, actual.site_counts)
        np.testing.assert_array_equal(expected.site_occupancy, actual.site_occupancy)
        for left, right in zip(expected.cluster_labels, actual.cluster_labels):
            np.testing.assert_array_equal(left, right)

    cases = [
        ("step1_detection", lambda: pick_origami_candidates(
            detection_points, bin_size_nm=8., connect_distance_nm=16., density_threshold=.02,
            minimum_points=5, candidate_template_points_nm=marks), compare_detection),
        ("step2_alignment", lambda: _align_regions_by_image_correlation(
            alignment_regions, rectangle_width_nm=65., rectangle_height_nm=45., requested_pixel_nm=1.,
            iterations=1, template_points_nm=small_grid, max_patch_pixels=128, sparse_pose_site_count=5,
            sparse_site_radius_nm=2.5, sparse_min_site_localizations=3), compare_alignment),
        ("step3_site_measurements", lambda: list(iter_sparse_site_evidence(
            regions, grid, site_radius_nm=2.5)), compare_sites),
        ("step3_and_step4_digital_classification", classify, compare_classification),
        ("fast_overlay", lambda: align_picked_origamis(
            regions, rows=8, columns=12, spacing_x_nm=10.909, spacing_y_nm=5.714,
            site_radius_nm=2.5, prealigned=True, preserve_pose=True, use_g5m=False,
            symmetrize_180=False), compare_overlay),
    ]
    report = dict(synthetic_only=True, warmup_excluded=True, seed=42,
                  candidates=args.candidates, sites_per_candidate=96,
                  localizations_per_candidate=2880,
                  gpu=(properties["name"].decode() if properties is not None else None),
                  vram_bytes=(properties["totalGlobalMem"] if properties is not None else None), cases={})
    for name, function, check in cases:
        outputs, timings = {}, {}
        for selected_mode in args.modes:
            print(f"{name}: {selected_mode}...", flush=True)
            outputs[selected_mode], timings[selected_mode] = measure(function, selected_mode, args.repeats)
            if selected_mode != "reference":
                check(outputs["reference"], outputs[selected_mode])
            print(f"  {timings[selected_mode]['median_seconds']:.4f}s; parity passed", flush=True)
        for selected_mode in args.modes[1:]:
            timings[selected_mode]["speedup_vs_reference"] = timings["reference"]["median_seconds"] / timings[selected_mode]["median_seconds"]
        report["cases"][name] = {"parity": "passed", "timings": timings}
    report["backend_status"] = compute.status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved synthetic benchmark: {args.output}", flush=True)


if __name__ == "__main__":
    main()
