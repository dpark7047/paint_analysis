"""Compare accelerated measurements with the untouched reference algorithms."""
import dataclasses
import os
import queue
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

import classification_compute as compute
from origami_analysis import (ideal_grid_points, iter_sparse_site_evidence,
                             sparse_site_evidence_diagnostics,
                             direct_digital_group_localization_evidence,
                             supported_site_centroids, supported_site_spacing_errors)


class CPUParity(unittest.TestCase):
    def test_hidden_and_windows_template_names_terminate(self):
        from paint_analysis_gui import custom_template_display_name
        self.assertEqual(custom_template_display_name("/tmp/.png"), ".png")
        self.assertEqual(custom_template_display_name("C:\\data\\template.png"), "template")
        self.assertEqual(custom_template_display_name("/tmp/:\\.png"), "\\")

    def test_progress_flood_yields_to_the_ui(self):
        from paint_analysis_gui import PaintAnalysisApp
        messages = queue.Queue()
        for index in range(100):
            messages.put(("status", str(index)))
        # Simulate nontrivial widget updates; draining the entire queue would
        # block Tk for half a second or more.
        app = SimpleNamespace(worker_queue=messages,
                              status=SimpleNamespace(set=lambda value: time.sleep(.005)),
                              after=mock.Mock(), _poll_worker=mock.Mock())
        start = time.monotonic()
        PaintAnalysisApp._poll_worker(app)
        self.assertLess(time.monotonic() - start, .25)
        self.assertGreater(messages.qsize(), 0)
        app.after.assert_called_once_with(100, app._poll_worker)

    def test_display_contrast_reuses_image_and_invalidates_changed_limits(self):
        from paint_analysis_gui import PaintAnalysisApp, scale_density_like_picasso
        render = dict(image=np.arange(16.).reshape(4, 4), min_density=0., max_density=15.)
        expected, limits = scale_density_like_picasso(render["image"], 0., 15.)
        actual, actual_limits = PaintAnalysisApp._prepare_origami_display_contrast(render)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_limits, limits)
        reused, _ = PaintAnalysisApp._prepare_origami_display_contrast(render)
        self.assertIs(actual, reused)
        render["max_density"] = 30.
        changed, _ = PaintAnalysisApp._prepare_origami_display_contrast(render)
        self.assertIsNot(actual, changed)
        np.testing.assert_array_equal(changed, scale_density_like_picasso(render["image"], 0., 30.)[0])

    def test_template_tree_cache_tracks_geometry_without_mutating_inputs(self):
        points = np.array([[0., 0.], [10., 0.]])
        with compute.use_mode("cpu"):
            first = compute.template_tree(points)
            self.assertIs(first, compute.template_tree(points.copy()))
            points[1, 0] = 20.
            second = compute.template_tree(points)
            self.assertIsNot(first, second)
            self.assertEqual(first.data[1, 0], 10.)
            self.assertEqual(second.data[1, 0], 20.)

    def test_vectorized_site_diagnostics_preserve_all_geometry_and_counts(self):
        rng = np.random.default_rng(781)
        grids = [ideal_grid_points(8, 12, 10.909, 5.714),
                 ideal_grid_points(2, 3, 12., 8.), np.array([[0., 0.]])]
        for grid in grids:
            for radius in (0.01, 1.5, 2.5, 8.):
                points = np.repeat(grid, 7, axis=0) + rng.normal(0, 1.5, (len(grid) * 7, 2))
                for region in (points, points[:1], np.empty((0, 2))):
                    with self.subTest(sites=len(grid), radius=radius, points=len(region)):
                        with compute.use_mode("reference"):
                            expected = sparse_site_evidence_diagnostics(region, grid, site_radius_nm=radius)
                        with compute.use_mode("cpu"):
                            actual = sparse_site_evidence_diagnostics(region, grid, site_radius_nm=radius)
                        for field in dataclasses.fields(expected):
                            np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name), err_msg=field.name)

    def test_chunked_nearest_preserves_exact_ties(self):
        points = np.array([[0., 0.], [1., 0.], [2., 0.]])
        grid = np.array([[-1., 0.], [1., 0.], [3., 0.]])
        with compute.use_mode("cpu"):
            distances, indices = compute.nearest_assignments(points, grid)
        np.testing.assert_array_equal(indices, [0, 1, 1])
        np.testing.assert_array_equal(distances, [1., 0., 1.])

    def test_centroid_reuse_preserves_spacing_errors(self):
        grid = ideal_grid_points(3, 4, 10., 8.)
        points = np.repeat(grid, 5, axis=0) + .2
        mask = np.ones(len(grid), dtype=bool)
        centroids = supported_site_centroids(points, grid, mask, site_radius_nm=2.)
        original = supported_site_spacing_errors(points, grid, mask, site_radius_nm=2.)
        with mock.patch("origami_analysis.supported_site_centroids", side_effect=AssertionError("repeated centroid calculation")):
            reused = supported_site_spacing_errors(points, grid, mask, site_radius_nm=2., centroids=centroids)
        self.assertEqual(original, reused)


class GPUParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with compute.use_mode("gpu"):
                compute.gpu_module()
        except Exception as exc:
            if os.environ.get("PAINT_REQUIRE_GPU_TESTS") == "1":
                raise
            raise unittest.SkipTest(str(exc))

    def test_batched_assignments_and_site_diagnostics(self):
        rng = np.random.default_rng(42)
        grid = ideal_grid_points(8, 12, 10.909, 5.714)
        regions = [np.repeat(grid, 12, axis=0) + rng.normal(0, 1., (len(grid)*12, 2)) for _ in range(35)]
        regions += [np.empty((0, 2)), np.array([grid[0] + [2.5, 0.]])]
        with compute.use_mode("reference"):
            reference = list(iter_sparse_site_evidence(regions, grid, site_radius_nm=2.5))
        with compute.use_mode("gpu"):
            actual = list(iter_sparse_site_evidence(regions, grid, site_radius_nm=2.5))
        for left, right in zip(reference, actual):
            for field in dataclasses.fields(left):
                np.testing.assert_allclose(getattr(right, field.name), getattr(left, field.name), rtol=1e-12, atol=1e-12, err_msg=field.name)

    def test_group_overlap_brightness_empty_candidates_and_radius_boundary(self):
        grid = ideal_grid_points(2, 3, 10., 8.)
        groups = [[0, 1, 2], [1, 2, 3], [4, 5]]
        rng = np.random.default_rng(3)
        regions = [np.repeat(grid, 15, axis=0) + rng.normal(0, .5, (90, 2)) for _ in range(34)]
        regions += [np.empty((0, 2)), np.array([grid[0] + [2., 0.]]), np.repeat(grid, 15000, axis=0)]
        kwargs = dict(assignment_radius_nm=2., brightness_factors=[2., .2, 1.])
        with compute.use_mode("reference"):
            reference = direct_digital_group_localization_evidence(regions, grid, groups, **kwargs)
        with compute.use_mode("gpu"):
            actual = direct_digital_group_localization_evidence(regions, grid, groups, **kwargs)
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-10)

    def test_large_density_filter_and_linear_fft_correlations(self):
        rng = np.random.default_rng(41)
        image = rng.random((512, 512))
        template = rng.random((9, 9))
        with compute.use_mode("gpu"):
            filtered = compute.gaussian_filter(image, sigma=1., mode="constant")
            correlation = compute.fftconvolve(image, template, mode="same")
        np.testing.assert_allclose(filtered, gaussian_filter(image, sigma=1., mode="constant"), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(correlation, fftconvolve(image, template, mode="same"), rtol=1e-12, atol=1e-12)

    def test_pose_decisions_keep_cpu_numerics_and_restore_gpu_context(self):
        @compute.cpu_only
        def fit():
            self.assertEqual(compute.mode(), "cpu")
            self.assertIsNone(compute.gpu_module())
            return 42
        with compute.use_mode("gpu"):
            self.assertEqual(fit(), 42)
            self.assertEqual(compute.mode(), "gpu")

    def test_auto_failure_retries_cpu_and_gpu_mode_is_explicit(self):
        with mock.patch.object(compute, "_failure", "test unavailable"):
            with compute.use_mode("auto"):
                self.assertIsNone(compute.gpu_module())
            with compute.use_mode("gpu"), self.assertRaisesRegex(RuntimeError, "test unavailable"):
                compute.gpu_module()


if __name__ == "__main__":
    unittest.main()
