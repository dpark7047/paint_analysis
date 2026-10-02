from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from origami_analysis import (
    _align_regions_by_image_correlation, _preserves_fiducial_support,
    alignment_fiducial_support,
)
from paint_analysis_gui import (
    PaintAnalysisApp, plot_alignment_pose_history, required_fiducial_mask,
    theoretical_grid_in_footprint,
)


SITES = np.array([[-30., -15.], [-30., 0.], [-30., 15.], [-20., 15.],
                  [30., -15.], [30., 15.]])


def support(points):
    return alignment_fiducial_support(points, SITES, 4., 3)


def test_one_bright_end_cannot_compensate_for_missing_other_end():
    points = np.repeat(SITES[:4], 500, axis=0)
    evidence = support(points)
    assert evidence['supported'].tolist() == [4, 0]
    assert evidence['required'].tolist() == [2, 1]
    assert not evidence['passed']
    # Individual missing marks are allowed when both ends have evidence.
    assert support(np.repeat(SITES[[0, 2, 4]], 3, axis=0))['passed']


def test_refinement_cannot_trade_fiducial_precision_for_interior_agreement():
    points = np.repeat(SITES, 3, axis=0)
    before = support(points)
    assert _preserves_fiducial_support(before, support(points + .05))
    assert not _preserves_fiducial_support(before, support(points + [1., 0.]))
    assert not _preserves_fiducial_support(before, support(points[:-6]))


def align_fixture(**kwargs):
    rng = np.random.default_rng(127)
    points = np.vstack([rng.normal(site, .3, (15, 2)) for site in SITES])
    points = np.vstack([points, rng.normal([0, 0], 1., (500, 2))])
    angle = np.deg2rad(27.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = points @ rotation.T + [200., 300.]
    image = np.zeros((80, 100))
    for x, y in SITES:
        image[int(y + 40), int(x + 50)] = 1
    history = []
    result = _align_regions_by_image_correlation(
        [points], rectangle_width_nm=100., rectangle_height_nm=80.,
        requested_pixel_nm=1., iterations=2, template_points_nm=SITES,
        template_image=gaussian_filter(image, 1.),
        # Mirrors Step 2, which disables later site-measurement gates.
        sparse_pose_site_count=0, sparse_site_radius_nm=4.,
        sparse_min_site_localizations=3, refinement_grid_points_nm=SITES,
        pose_history_output=history, **kwargs)
    return points, result, history


@pytest.mark.parametrize('stage,function,return_tail', [
    ('Whole-lattice refinement', '_refine_pose_by_full_lattice', (100.,)),
    ('Lattice-centroid refinement', '_refine_pose_from_lattice_centroids', (6, 0.)),
])
def test_dense_interior_fit_rejects_harmful_refinement_and_records_attempt(stage, function, return_tail):
    with patch('origami_analysis.' + function, return_value=(np.eye(2), np.array([20., 0.]), *return_tail)):
        points, result, history = align_fixture()
    aligned, centers, corners, angles, *_ = result
    assert support(aligned[0])['passed']
    assert abs((angles[0] - 27 + 180) % 360 - 180) < 1.
    attempted = next(entry for entry in history[0] if entry['stage'] == stage)
    assert not attempted['accepted']
    assert history[0][-1]['passed']
    # The inspection overlay must use the exact inverse of the fitted pose.
    np.testing.assert_allclose(theoretical_grid_in_footprint(SITES, corners[0]),
                               history[0][-1]['world_sites'], atol=1e-10)
    world_sites = history[0][-1]['world_sites']
    x_axis = (corners[0, 1] - corners[0, 0]) / 100.
    y_axis = (corners[0, 3] - corners[0, 0]) / 80.
    np.testing.assert_allclose((world_sites - corners[0].mean(axis=0)) @ np.array([x_axis, y_axis]).T,
                               SITES, atol=1e-10)
    figure = Figure()
    plot_alignment_pose_history(figure, SimpleNamespace(
        regions=[points], alignment_pose_history=history), 0)
    assert len([axis for axis in figure.axes if axis.get_visible()]) == 5
    assert any('DISCARDED' in axis.get_title() for axis in figure.axes)


def test_gui_alignment_gate_rejects_missing_end_even_with_correlation_gate_off():
    picks = SimpleNamespace(
        aligned_regions=[np.repeat(SITES[:4], 500, axis=0), np.repeat(SITES, 3, axis=0)],
        template_points_nm=SITES, point_counts=np.array([2000, 18]),
        rectangle_confidence=np.array([.99, .99]),
    )
    params = dict(alignment_template_image=np.ones((2, 2)),
                  alignment_template_overlay_points_nm=SITES,
                  site_mask_radius_nm=4., min_site_localizations=3,
                  use_correlation_gate=False, require_corner_support=False)
    np.testing.assert_array_equal(required_fiducial_mask(picks, params), [False, True])
    with patch.object(PaintAnalysisApp, '_reject_overlapping_origami_fits',
                      side_effect=lambda p, settings, accepted: accepted):
        np.testing.assert_array_equal(
            PaintAnalysisApp._apply_origami_alignment_filters(picks, params), [False, True])
