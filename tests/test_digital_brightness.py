"""Per-group calibration must agree in measurement, inspection, and classification."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from origami_analysis import direct_digital_group_localization_evidence
from paint_analysis_gui import (
    PaintAnalysisApp, logical_bit_model_from_metadata, origami_grid_points,
    digital_group_decision_annotations, exact_digital_template_matches,
    spatial_digital_pixel_params,
)


def metadata():
    return {"rows": 1, "columns": 3, "logical_model": {
        "format": "paint-analysis-logical-bits-v1", "physical_rows": 1,
        "physical_columns": 3, "active_logical_bits": ["A", "B", "C"],
        "logical_bits": [
            {"id": "B", "physical_sites": [[1, 2]], "brightness_factor": 0.2},
            {"id": "A", "physical_sites": [[1, 1]], "brightness_factor": 2},
            {"id": "C", "physical_sites": [[1, 3]]},
        ]}}


class BrightnessTests(unittest.TestCase):
    def test_schema_sorting_defaults_and_validation(self):
        model = logical_bit_model_from_metadata(metadata())
        self.assertEqual(model["bit_ids"], ("A", "B", "C"))
        self.assertEqual(model["bit_brightness_factors"], (2.0, 0.2, 1.0))
        for invalid in (0, -1, float("nan"), float("inf"), True, "2", None):
            value = metadata()
            value["logical_model"]["logical_bits"][0]["brightness_factor"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "brightness_factor"):
                logical_bit_model_from_metadata(value)

    def test_measurement_divides_by_size_and_brightness_without_changing_overlap(self):
        grid = np.array([[0., 0.], [10., 0.]])
        points = np.repeat(grid, 10, axis=0)
        measured = direct_digital_group_localization_evidence(
            [points], grid, [[0, 1], [0, 1]], assignment_radius_nm=1,
            brightness_factors=[2, 0.2])
        np.testing.assert_allclose(measured, [[5, 50]])
        for factors in ([1], [0, 1], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                direct_digital_group_localization_evidence(
                    [points], grid, [[0], [1]], assignment_radius_nm=1,
                    brightness_factors=factors)

    def test_scaled_signal_matches_normal_calls_and_step4(self):
        model = logical_bit_model_from_metadata(metadata())
        params = dict(rows=1, columns=3, spacing_x_nm=20., spacing_y_nm=20.,
                      site_mask_radius_nm=2., min_site_localizations=3,
                      min_site_evidence=0.1, digital_pixel_model=model, logical_model=model)
        grid = origami_grid_points(1, 3, 20., 20., params)
        # Raw support A=10, B=1, C=5. Each is equivalent to support=5.
        points = np.concatenate([np.repeat(grid[list(cells)], n, axis=0)
                                 for cells, n in zip(model["bit_physical_cells"], [10, 1, 5])])
        picks = SimpleNamespace(aligned_regions=[points])
        PaintAnalysisApp._measure_direct_digital_groups(picks, params)
        np.testing.assert_allclose(params["digital_group_localization_evidence"], [[5, 5, 5]])
        self.assertTrue(exact_digital_template_matches(params, 1)[0])
        annotations = digital_group_decision_annotations(grid, params, 0, include_statistics=True)
        self.assertTrue(all(" ON " in item[1] for item in annotations))
        self.assertIn("brightness factor=0.2", annotations[1][1])
        # The brighter group must fail when it has only the normal amount.
        normal_points = np.concatenate([np.repeat(grid[list(cells)], n, axis=0)
                                        for cells, n in zip(model["bit_physical_cells"], [5, 1, 5])])
        dim_a = copy.deepcopy(params)
        PaintAnalysisApp._measure_direct_digital_groups(SimpleNamespace(aligned_regions=[normal_points]), dim_a)
        self.assertFalse(exact_digital_template_matches(dim_a, 1)[0])
        self.assertEqual(dim_a["classification_observed_digital_states"], ((False, True, True),))
        unscaled = copy.deepcopy(params)
        unscaled["digital_pixel_model"]["bit_brightness_factors"] = (1., 1., 1.)
        PaintAnalysisApp._measure_direct_digital_groups(picks, unscaled)
        self.assertFalse(exact_digital_template_matches(unscaled, 1)[0])

    def test_aggregate_replacement_keeps_factor_order(self):
        model = dict(bit_ids=("full_align", "B"), bit_physical_cells=((0, 1), (1,)),
                     bit_brightness_factors=(4., 0.2), physical_shape=(1, 2))
        result = spatial_digital_pixel_params({"digital_pixel_model": model},
                                             SimpleNamespace(accepted_mask=np.array([True])))
        self.assertEqual(result["digital_pixel_model"]["bit_brightness_factors"], (0.2, 1.))
