from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from origami_analysis import explicit_dark_geometry, dark_mask_contains, pick_origami_candidates
from paint_analysis_gui import (dark_groups_from_metadata, logical_bit_model_from_metadata,
    origami_alignment_dark_groups, mirror_origami_template_inputs, PaintAnalysisApp)


def metadata():
    return dict(rows=3, columns=4, logical_model=dict(
        format='paint-analysis-logical-bits-v1', physical_rows=3, physical_columns=4,
        logical_bits=[], active_logical_bits=[], alignment_groups=[dict(id='fid', physical_sites=[[1, 1], [3, 4]])],
        dark_groups=[dict(id='empty', physical_sites=[[1, 2], [2, 3]], mask_radius_nm=2.)]))


def test_dark_metadata_is_not_a_barcode_bit_and_respects_mirroring():
    model = logical_bit_model_from_metadata(metadata())
    assert model['bit_ids'] == ()
    assert model['dark_groups'][0]['cells'] == (5, 10)
    params = dict(rows=3, columns=4, spacing_x_nm=10., spacing_y_nm=5., digital_pixel_model=model)
    original = origami_alignment_dark_groups(params)
    mirrored = origami_alignment_dark_groups(mirror_origami_template_inputs(params))
    # Cell order reverses within rows; compare sets of physical coordinates.
    expected = {tuple(point) for point in np.asarray(original[0]['points_nm']) * [-1, 1]}
    assert set(mirrored[0]['points_nm']) == expected
    assert mirrored[0]['mask_radius_nm'] == 2.


def test_explicit_mask_has_only_declared_circles_and_no_inferred_gaps():
    groups = [dict(points_nm=[[0., 0.]], mask_radius_nm=2.),
              dict(points_nm=[[10., 0.]], mask_radius_nm=1.)]
    mask = explicit_dark_geometry(groups)
    np.testing.assert_array_equal(dark_mask_contains([[0, 0], [1.9, 0], [5, 0], [11.1, 0]], mask), [1, 1, 0, 0])
    assert explicit_dark_geometry(()) is None
    assert origami_alignment_dark_groups({'digital_pixel_model': {'bit_physical_cells': ((0,),)}}) == ()


def test_step1_receives_explicit_masks_and_cache_tracks_radius():
    model = logical_bit_model_from_metadata(metadata())
    shared = dict(rows=3, columns=4, physical_shape=(3, 4), spacing_x_nm=10., spacing_y_nm=5.,
                  dark_groups=model['dark_groups'], overlay_points_nm=np.array([[-15., -5.], [15., 5.]]))
    var = lambda value: SimpleNamespace(get=lambda: value)
    app = SimpleNamespace(origami_source_points_nm=np.array([[0., 0.], [1., 1.]]),
        origami_source_candidate_fingerprint='', origami_pick_bin_nm=var(5.),
        origami_connect_distance_nm=var(15.), origami_signal_gap_nm=var(20.),
        origami_min_density_contrast=var(.2), origami_min_points=var(1), origami_shared_alignment_template=shared)
    app._origami_candidate_template_points = lambda: PaintAnalysisApp._origami_candidate_template_points(app)
    before = PaintAnalysisApp._origami_candidate_stage_signature(app)
    groups = PaintAnalysisApp._origami_candidate_dark_groups(app)
    with patch('origami_analysis._pick_template_supported_regions', return_value='picked') as picker:
        assert pick_origami_candidates(app.origami_source_points_nm, bin_size_nm=5., connect_distance_nm=15.,
            density_threshold=.2, candidate_template_points_nm=shared['overlay_points_nm'],
            alignment_dark_groups_nm=groups) == 'picked'
    assert picker.call_args.kwargs['alignment_dark_groups_nm'] == groups
    shared['dark_groups'][0]['mask_radius_nm'] = 4.
    assert PaintAnalysisApp._origami_candidate_stage_signature(app) != before


@pytest.mark.parametrize('radius', [-1, 0, float('nan'), True])
def test_parser_rejects_invalid_dark_radius(radius):
    source = metadata()
    source['logical_model']['dark_groups'][0]['mask_radius_nm'] = radius
    with pytest.raises(ValueError, match='radius'):
        dark_groups_from_metadata(source)


def test_detection_template_keeps_dark_metadata_without_classification_templates():
    from paint_analysis_gui import origami_with_shared_alignment, alignment_template_overlay_points
    shared = dict(image=np.eye(3), physical_shape=(3, 4), rows=3, columns=4,
                  spacing_x_nm=10., spacing_y_nm=5., dark_groups=dark_groups_from_metadata(metadata()),
                  overlay_points_nm=np.array([[-15., -5.], [15., 5.]]))
    params = dict(rows=3, columns=4, spacing_x_nm=10., spacing_y_nm=5.,
                  alignment_template_image=shared['image'])
    assert origami_alignment_dark_groups(params) == ()
    fitted = origami_with_shared_alignment(params, shared)
    assert origami_alignment_dark_groups(fitted)
    assert 'custom_templates' not in fitted
    assert fitted['alignment_template_image'] is not shared['image']
    mirrored = mirror_origami_template_inputs(fitted)
    expected = np.asarray(origami_alignment_dark_groups(fitted)[0]['points_nm']) * [-1, 1]
    actual = origami_alignment_dark_groups(mirrored)[0]['points_nm']
    assert set(map(tuple, expected)) == set(actual)
    np.testing.assert_allclose(alignment_template_overlay_points(np.empty((0, 2)), mirrored),
                               shared['overlay_points_nm'] * [-1, 1])
