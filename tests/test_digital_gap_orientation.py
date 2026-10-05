from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from origami_analysis import _align_regions_by_image_correlation
from paint_analysis_gui import (
    ORIGAMI_ALIGNMENT_CACHE_KEYS, origami_digital_alignment_cells,
    origami_stage_cache_matches, plot_alignment_pose_history,
)


DIGITAL = np.array([[-14., -8.], [-14., 8.], [-4., -8.], [-4., 8.]])
FIDUCIALS = np.array([[-30., -15.], [-30., 15.], [30., -15.], [30., 15.]])


def run_alignment(initial_angle, *, dense_gaps=True, missing_reversed_end=False, both_dark=False):
    rng = np.random.default_rng(217)
    sites = FIDUCIALS.copy()
    if missing_reversed_end:
        sites[0, 1] = -5.  # Removes 180° symmetry of fiducial support.
        sites[1, 1] = 5.
    points = np.vstack([rng.normal(site, .15, (30, 2)) for site in sites])
    points = np.vstack([points, np.repeat(DIGITAL, 30, axis=0)])
    if dense_gaps:
        points = np.vstack([points, rng.uniform([4., -8.], [14., 8.], (500, 2))])
    image = np.zeros((60, 80))
    for x, y in sites:
        image[int(y + 30), int(x + 40)] = 1
    history = []
    with patch('origami_analysis._full_pose_rotation_candidates', return_value=[initial_angle]):
        result = _align_regions_by_image_correlation(
            [points], rectangle_width_nm=80., rectangle_height_nm=60.,
            requested_pixel_nm=1., iterations=1, template_points_nm=sites,
            template_image=gaussian_filter(image, 1.),
            sparse_site_radius_nm=2., sparse_min_site_localizations=3,
            digital_site_points_nm=DIGITAL, pose_history_output=history,
            alignment_dark_groups_nm=({"points_nm": [[-9., 0.], [9., 0.]] if both_dark else [[-9., 0.]], "mask_radius_nm": 3.},),
            refinement_grid_points_nm=np.vstack([sites, DIGITAL]),
        )
    return points, result, history


def test_failed_dark_check_triggers_bright_reversal_then_second_dark_check():
    points, result, history = run_alignment(180.)
    checks = [entry for entry in history[0] if entry['stage'].startswith('Dark check')]
    assert len(checks) == 2
    assert not checks[0]['accepted']
    assert checks[1]['accepted'] and checks[1]['bright_passed']
    assert checks[1]['dark_bright_ratio'] <= checks[1]['max_dark_bright_ratio']
    aligned, centers, corners, angles, *_ = result
    assert abs((angles[0]+180)%360-180) < 1.
    axes = np.array([(corners[0, 1]-corners[0, 0])/80., (corners[0, 3]-corners[0, 0])/60.])
    np.testing.assert_allclose((history[0][-1]['world_sites']-corners[0].mean(axis=0)) @ axes.T, FIDUCIALS, atol=1e-9)
    figure = Figure()
    plot_alignment_pose_history(figure, SimpleNamespace(regions=[points], alignment_pose_history=history), 0)
    assert any('dark/bright:' in axis.get_title() for axis in figure.axes)


def test_passing_original_dark_check_never_retries_or_flips():
    with patch('origami_analysis._retry_reversed_bright_pose', side_effect=AssertionError('unexpected retry')):
        _, result, history = run_alignment(0.)
    checks = [entry for entry in history[0] if entry['stage'].startswith('Dark check')]
    assert len(checks) == 1 and checks[0]['accepted']
    assert abs((result[3][0]+180)%360-180) < 1.


def test_both_dark_checks_fail_without_silently_accepting_the_alternative():
    _, result, history = run_alignment(180., both_dark=True)
    checks = [entry for entry in history[0] if entry['stage'].startswith('Dark check')]
    assert len(checks) == 2 and not any(entry['accepted'] for entry in checks)
    assert abs((result[3][0]-180+180)%360-180) < 1.


def test_retry_can_correct_small_angular_and_translation_errors():
    from origami_analysis import _retry_reversed_bright_pose, alignment_fiducial_support
    sites = np.vstack([FIDUCIALS, [-25., 5.]])
    angle = np.deg2rad(185.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.repeat(sites, 20, axis=0) @ rotation.T + [4., -3.]
    valid = lambda p: alignment_fiducial_support(p, sites, 2., 3)['passed']
    aligned, correction, offset, passed = _retry_reversed_bright_pose(points, sites, 2., 3, 5, valid)
    assert passed
    np.testing.assert_allclose(aligned, np.repeat(sites, 20, axis=0), atol=.05)
    assert np.all(np.abs(offset) <= 10.+1e-8)
    assert abs(np.rad2deg(np.arctan2(-correction[1,0], -correction[0,0]))) <= 10.+1e-8


def test_dark_check_normalizes_density_and_rejects_missing_bright_support():
    from origami_analysis import explicit_dark_geometry, dark_alignment_ratio
    geometry = explicit_dark_geometry([dict(points_nm=[[0.,0.]], mask_radius_nm=2.)])
    sites = np.array([[10., 0.], [-10., 0.]])
    bright = np.repeat(sites, 20, axis=0)
    assert dark_alignment_ratio(bright, sites, geometry, 2., 3) == 0
    points = np.vstack([bright, np.zeros((10,2))])
    assert abs(dark_alignment_ratio(points, sites, geometry, 2., 3)-.5) < 1e-6
    assert np.isinf(dark_alignment_ratio(np.zeros((10,2)), sites, geometry, 2., 3))


def test_digital_schema_changes_invalidate_alignment_cache():
    old = {'digital_pixel_model': {'bit_physical_cells': ((0, 1),)}}
    new = {'digital_pixel_model': {'bit_physical_cells': ((2, 3),)}}
    assert not origami_stage_cache_matches(old, new, ORIGAMI_ALIGNMENT_CACHE_KEYS)
    assert origami_digital_alignment_cells(new) == ((2, 3),)
    assert origami_digital_alignment_cells({}) is None



def test_dark_gate_rejects_failed_pose_and_threshold_changes_invalidate_cache():
    from paint_analysis_gui import required_dark_mask
    sites = np.array([[10., 0.], [-10., 0.]])
    bright = np.repeat(sites, 20, axis=0)
    picks = SimpleNamespace(point_counts=np.array([40, 50]), template_points_nm=sites,
                            aligned_regions=[bright, np.vstack([bright, np.zeros((10, 2))])])
    params = dict(alignment_template_image=np.eye(3), site_mask_radius_nm=2.,
                  min_site_localizations=3, max_dark_bright_ratio=.25)
    groups = [dict(points_nm=[[0., 0.]], mask_radius_nm=2.)]
    with patch('paint_analysis_gui.origami_alignment_dark_groups', return_value=groups), \
         patch('paint_analysis_gui.alignment_template_overlay_points', return_value=sites):
        np.testing.assert_array_equal(required_dark_mask(picks, params), [True, False])
        assert not required_dark_mask(picks, params, 1)[0]
        assert required_dark_mask(picks, dict(params, max_dark_bright_ratio=.6), 1)[0]
    assert not origami_stage_cache_matches(params, dict(params, max_dark_bright_ratio=.6), ORIGAMI_ALIGNMENT_CACHE_KEYS)


def test_dark_retry_keeps_reference_pose_and_stays_on_cpu_in_gpu_mode():
    import classification_compute as compute
    with compute.use_mode("reference"):
        _, expected, expected_history = run_alignment(180.)
    with compute.use_mode("gpu"), patch.object(compute, "gpu_module", side_effect=AssertionError("GPU pose selection")):
        _, actual, actual_history = run_alignment(180.)
        assert compute.mode() == "gpu"
    for left, right in zip(expected[:4], actual[:4]):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-10)
    for left, right in zip(expected_history[0], actual_history[0]):
        assert left['stage'] == right['stage']
        assert left['accepted'] == right['accepted']
        if 'dark_bright_ratio' in left:
            assert left['dark_bright_ratio'] == right['dark_bright_ratio']
