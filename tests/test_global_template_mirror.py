from types import SimpleNamespace

import numpy as np

from paint_analysis_gui import (
    PaintAnalysisApp, mirror_origami_template_inputs, origami_grid_points,
    alignment_template_overlay_points, classified_template_overlay_points,
    exact_digital_template_matches, origami_stage_cache_matches,
    ORIGAMI_ALIGNMENT_CACHE_KEYS, independent_origami_pipeline_params,
)


def inputs():
    model = dict(physical_shape=(2, 3), bit_ids=('A', 'B'),
                 bit_cells=((0, 3), (2,)), bit_physical_cells=((0, 3), (2,)),
                 alignment_cells=(1, 5), active_bits=(True, False),
                 bit_brightness_factors=(2., 0.2), column_offsets_nm=(0., 2., 7.))
    image = np.arange(15).reshape(3, 5)
    template = dict(image=image, logical_model=model, column_offsets_nm=(0., 2., 7.),
                    rows=2, columns=3, spacing_x_nm=20., spacing_y_nm=20.,
                    overlay_points_nm=np.array([[-20., -10.], [27., 10.]]))
    return dict(rows=2, columns=3, spacing_x_nm=20., spacing_y_nm=20.,
                column_offsets_nm=(0., 2., 7.), digital_pixel_model=model,
                logical_model=model, shared_alignment_template=template,
                custom_templates=[template], alignment_template_image=image,
                mirror_all_templates=True, site_mask_radius_nm=2.,
                min_site_localizations=3, min_site_evidence=0.1)


def test_rasters_groups_offsets_and_overlays_reflect_together_without_mutation():
    original = inputs()
    mirrored = mirror_origami_template_inputs(original)
    for raster in (mirrored['alignment_template_image'],
                   mirrored['shared_alignment_template']['image'],
                   mirrored['custom_templates'][0]['image']):
        np.testing.assert_array_equal(raster, original['alignment_template_image'][:, ::-1])
    np.testing.assert_array_equal(original['alignment_template_image'], np.arange(15).reshape(3, 5))
    assert mirrored['column_offsets_nm'] == (-7., -2., 0.)
    for model in (mirrored['digital_pixel_model'], mirrored['logical_model'],
                  mirrored['custom_templates'][0]['logical_model']):
        assert model['bit_physical_cells'] == ((2, 5), (0,))
        assert model['alignment_cells'] == (1, 3)
        assert model['bit_ids'] == ('A', 'B')
        assert model['bit_brightness_factors'] == (2., 0.2)
        assert model['active_bits'] == (True, False)
    assert original['digital_pixel_model']['bit_physical_cells'] == ((0, 3), (2,))
    grid = origami_grid_points(2, 3, 20., 20., original)
    mirrored_grid = origami_grid_points(2, 3, 20., 20., mirrored)
    for old_cells, new_cells in zip(original['digital_pixel_model']['bit_cells'],
                                    mirrored['digital_pixel_model']['bit_cells']):
        np.testing.assert_allclose(mirrored_grid[list(new_cells)], grid[list(old_cells)] * [-1, 1])
    np.testing.assert_allclose(alignment_template_overlay_points(grid, mirrored),
                               alignment_template_overlay_points(grid, original) * [-1, 1])
    expected_overlay = classified_template_overlay_points(grid, original) * [-1, 1]
    actual_overlay = classified_template_overlay_points(mirrored_grid, mirrored)
    assert set(map(tuple, actual_overlay)) == set(map(tuple, expected_overlay))
    restored = mirror_origami_template_inputs(mirrored)
    assert restored['digital_pixel_model'] == original['digital_pixel_model']
    np.testing.assert_array_equal(restored['alignment_template_image'], original['alignment_template_image'])


def test_candidate_detection_tracks_toggle_and_does_not_accumulate_flips():
    params = inputs()
    app = PaintAnalysisApp.__new__(PaintAnalysisApp)
    app.origami_shared_alignment_template = params['shared_alignment_template']
    app.origami_mirror_all_templates = SimpleNamespace(get=lambda: False)
    normal = app._origami_candidate_template_points()
    app.origami_mirror_all_templates = SimpleNamespace(get=lambda: True)
    np.testing.assert_array_equal(app._origami_candidate_template_points(), normal * [-1, 1])
    np.testing.assert_array_equal(app._origami_candidate_template_points(), normal * [-1, 1])
    app.origami_mirror_all_templates = SimpleNamespace(get=lambda: False)
    np.testing.assert_array_equal(app._origami_candidate_template_points(), normal)


def test_measurement_and_classification_use_mirrored_groups_and_keep_brightness():
    original = inputs()
    grid = origami_grid_points(2, 3, 20., 20., original)
    points = np.repeat(grid[[0, 3]], 12, axis=0)
    PaintAnalysisApp._measure_direct_digital_groups(SimpleNamespace(aligned_regions=[points]), original)
    assert exact_digital_template_matches(original, 1)[0]
    mirrored = mirror_origami_template_inputs(inputs())
    PaintAnalysisApp._measure_direct_digital_groups(SimpleNamespace(aligned_regions=[points * [-1, 1]]), mirrored)
    np.testing.assert_allclose(mirrored['digital_group_localization_evidence'], original['digital_group_localization_evidence'])
    assert exact_digital_template_matches(mirrored, 1)[0]


def test_orientation_change_invalidates_alignment_cache_and_survives_tiling():
    old = dict(mirror_all_templates=False)
    new = dict(mirror_all_templates=True)
    assert not origami_stage_cache_matches(old, new, ORIGAMI_ALIGNMENT_CACHE_KEYS)
    params = mirror_origami_template_inputs(inputs())
    params['_prealigned_picks'] = object()
    tile = independent_origami_pipeline_params(params)
    assert tile['mirror_all_templates']
    assert '_prealigned_picks' not in tile
    np.testing.assert_array_equal(tile['alignment_template_image'], params['alignment_template_image'])
    assert tile['digital_pixel_model'] == params['digital_pixel_model']
