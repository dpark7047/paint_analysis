from pathlib import Path
import pytest
from analysis_session import save_analysis_session, load_analysis_session
import ast, types, tkinter as tk
from unittest.mock import Mock, call
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import paint_analysis_gui as gui


@pytest.fixture(autouse=True)
def restore_tk_default_root():
    previous = tk._default_root
    yield
    tk._default_root = previous


class App:
    def __getattr__(self, key):
        fn = gui.PaintAnalysisApp.__dict__.get(key)
        if isinstance(fn, staticmethod): return fn.__func__
        if callable(fn): return types.MethodType(fn, self)
        raise AttributeError(key)

def app():
    master=tk.Tcl();tk._default_root=master
    result=App();result.master_interpreter=master
    tree=ast.parse(Path(gui.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PaintAnalysisApp')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignments=[n for n in init.body if isinstance(n,(ast.Assign,ast.AnnAssign))]
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'<init>','exec'),dict(vars(gui),self=result))
    result.origami_figure=Figure(figsize=(10,7));result.origami_canvas=FigureCanvasAgg(result.origami_figure)
    for name in ('origami_toolbar','notebook','origami_match_panel_combo','origami_template_result_combo','origami_review_combo'):
        setattr(result,name,Mock())
    result._configure_origami_navigation_controls=Mock()
    result._refresh_origami_action_states=Mock()
    result.update_idletasks=lambda:None
    result.after_idle=lambda f: None
    result.after=lambda *args:None
    result._schedule_origami_zoom_render=lambda *args:None
    result._after_load=lambda *args,**kwargs:None
    result._refresh_filter_list=Mock();result._update_filter_bounds_label=Mock();result._update_roi_label=Mock()
    result._show_origami_stage=Mock();result._finish_origami_identification_progress=Mock()
    return result

def populate(a):
    from dataclasses import replace
    import pandas as pd
    from pathlib import Path
    from origami_analysis import identify_origami_regions, ideal_grid_points, concatenate_origami_pick_results, _render_candidate_image
    grid=ideal_grid_points(2,2,10,10)
    rng=np.random.default_rng(12)
    base_points=np.vstack([rng.normal(p,.15,(8,2)) for p in grid])
    base=identify_origami_regions(base_points,pick_bin_size_nm=3,connect_distance_nm=15,density_threshold=0,min_candidate_points=1,max_candidate_points=1000,rows=2,columns=2,spacing_x_nm=10,spacing_y_nm=10,measure_sites=False,compute_grid_blob_bic=False)
    assert len(base.regions)==1
    candidates=[]
    for i in range(36):
        cells=([0,2] if i%3==0 else [0,1,2,3] if i%3==1 else [1,3])
        local=np.vstack([rng.normal(grid[j],.15,(8,2)) for j in cells])
        shift=np.array([i*100.,0.])
        corners=np.array([[-15.,-15.],[15.,-15.],[15.,15.],[-15.,15.]])+shift
        candidates.append(replace(base,regions=[local+shift],aligned_regions=[local],accepted_mask=np.array([True]),
            point_counts=np.array([len(local)]),rectangle_corners_nm=corners[None],
            template_points_nm=grid,alignment_reference_image=_render_candidate_image(base_points,np.zeros(2),40,1,1),alignment_pixel_nm=1.,alignment_canvas_side_nm=40.))
    picks=concatenate_origami_pick_results(candidates)
    model=dict(bit_ids=('a','b'),bit_cells=((0,2),(1,3)),bit_physical_cells=((0,2),(1,3)),active_bits=(True,False),physical_shape=(2,2),bit_brightness_factors=(1.,1.))
    params=dict(rows=2,columns=2,spacing_x_nm=10.,spacing_y_nm=10.,rectangle_margin_nm=10.,site_mask_radius_nm=2.,min_site_localizations=3,min_site_evidence=.1,
        digital_pixel_model=model,logical_model=model,classification_method='exact digital ON/OFF lookup',identification_generation=7,
        classification_lookup_eligible=(True,)*36,classification_qc_eligible=(True,)*36,alignment_template_image=np.eye(5),
        alignment_template_overlay_points_nm=tuple(map(tuple,grid)),template_pixel_size_x_nm=1.,template_pixel_size_y_nm=1.,
        _inspection_stage=5,min_candidate_points=1,max_candidate_points=1000,min_supported_sites=0,min_supported_rows=0,min_supported_columns=0,
        min_rectangle_confidence=0.,use_correlation_gate=False,max_site_spacing_error_nm=float('inf'))
    a._measure_direct_digital_groups(picks,params)
    for name,bits in [('A',(True,False)),('Full',(True,True))]:
        p=dict(params,logical_model=dict(model,active_bits=bits))
        mask=gui.exact_digital_template_matches(p,36)
        p['classification_dispositions']=tuple('classified as '+name if v else 'unclassified: no template passed' for v in mask)
        a.origami_multi_template_results[name]=dict(picks=replace(picks,accepted_mask=mask),params=p)
        a.origami_multi_template_counts[name]=int(mask.sum())
    a.origami_pick_result=a.origami_multi_template_results['A']['picks'];a.origami_identification_params=a.origami_multi_template_results['A']['params']
    a.origami_identification_generation=7
    a.origami_source_points_nm=np.vstack(picks.regions)
    a.origami_source_locs=pd.DataFrame(a.origami_source_points_nm,columns=['x','y'])
    a.loaded=gui.LoadedData(Path('/missing/source.csv'),a.origami_source_locs,[{'Pixelsize':1}],{})
    a.corrected_locs=a.origami_source_locs
    a.origami_loaded_source_path=a.loaded.path
    a.origami_multi_template_unclassified_count=12
    a.origami_multi_template_unclassified_centers_nm=np.array([np.median(picks.regions[i],axis=0) for i in range(2,36,3)])
    a.origami_multi_template_unclassified_details=[]
    a.origami_gallery_page_size.set('16')
    def work(fn):
        kind,payload=fn()
        if kind=='origami': a._plot_origami_analysis(payload)
        else: raise ValueError(kind)
    a._run_worker=work
    return a

VIEWS = ['Coarse identification density', 'Identified origami template matches',
    'Origami type counts', 'Open vs closed', 'Digital-group bias heatmap',
    'Digital-group threshold audit', 'Unmatched-pattern audit', 'Unclassified evidence distributions',
    'Threshold sensitivity',
    'Digital-pixel spatial heatmap', 'Individual origami gallery', 'Individual site assignments',
    'Selected origami detail', 'Digital bit derivation', 'Aligned density',
    'Integrated density per site', 'Mean site counts', 'Site occupancy']


def snapshot(a):
    result = []
    for axis in a.origami_figure.axes:
        result.append((axis.get_title(), axis.get_xlabel(), axis.get_ylabel(),
                       [text.get_text() for text in axis.texts],
                       [text.get_text() for text in axis.get_xticklabels()],
                       [text.get_text() for text in axis.get_yticklabels()]))
        result.extend(np.asarray(image.get_array()).copy() for image in axis.images)
        result.extend(np.asarray(line.get_xydata()).copy() for line in axis.lines)
        result.extend(np.asarray(collection.get_offsets()).copy() for collection in axis.collections)
    return result


def matrix(a, monkeypatch):
    snapshots = {}
    notices = []
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *args, **kwargs: notices.append(args))
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *args, **kwargs: pytest.fail(str(args)))
    for name in ['All templates', 'A', 'Full', 'Unclassified']:
        a.origami_template_result_view.set(name)
        a.origami_plot_option.set('Origami type counts')
        a._on_origami_template_result_selection()
        for view in (VIEWS[:10] if name == 'All templates' else VIEWS):
            notices.clear()
            a.origami_plot_option.set(view)
            if view in ('Selected origami detail', 'Digital bit derivation'):
                a.origami_selected_index = 0
            a.render_origami_plot()
            a.origami_canvas.draw()
            if notices:
                # Whole-image results have no single coarse grid even live.
                assert view == 'Coarse identification density', (name, view, notices)
                assert notices[0][0] == 'Tile-local coarse maps'
                snapshots[name, view] = list(notices)
            else:
                assert a.origami_last_rendered_plot_option == view, (name, view, a.status.get())
                snapshots[name, view] = snapshot(a)
    return snapshots


def test_all_classification_views_match_after_archive_round_trip(tmp_path, monkeypatch):
    live = populate(app())
    before = matrix(live, monkeypatch)
    saved_class = live.origami_template_result_view.get()
    saved_view = live.origami_plot_option.get()
    payload = live._capture_origami_analysis()
    assert set(payload['diagnostics']) == {'origami_latest_sweep'}
    path = tmp_path / 'views.paintanalysis'
    save_analysis_session(path, payload)
    restored = app()
    loaded = load_analysis_session(path, live._analysis_record_types())
    restored._install_origami_analysis(loaded)
    assert restored.origami_template_result_view.get() == saved_class
    assert restored.origami_plot_option.get() == saved_view
    restored._run_worker = lambda fn: pytest.fail('Recomputed a saved overlay or diagnostic')
    after = matrix(restored, monkeypatch)
    assert before.keys() == after.keys()
    for key in before:
        assert len(before[key]) == len(after[key]), key
        for original, reopened in zip(before[key], after[key]):
            if isinstance(original, np.ndarray):
                np.testing.assert_allclose(original, reopened, equal_nan=True, err_msg=str(key))
            else:
                assert original == reopened, key

@pytest.mark.parametrize('selected', ['A', 'Full', 'Unclassified'])
@pytest.mark.parametrize('view', ['Aligned density', 'Individual origami gallery', 'Site occupancy'])
def test_older_archives_build_missing_overlays_for_each_class(tmp_path, monkeypatch, selected, view):
    live = populate(app())
    live.origami_template_result_view.set(selected)
    live.origami_plot_option.set(view)
    payload = live._capture_origami_analysis()
    payload.pop('diagnostics', None)
    for key in ('origami_selected_index', 'origami_selected_match_index', 'origami_gallery_view_limits'):
        payload['state'].pop(key, None)
    path = tmp_path / 'older.paintanalysis'
    save_analysis_session(path, payload)
    restored = app()
    jobs = []
    def worker(fn):
        kind, result = fn()
        jobs.append(kind)
        assert kind == 'origami'
        assert result['identification_generation'] == restored.origami_identification_generation
        restored._plot_origami_analysis(result)
    restored._run_worker = worker
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *args, **kwargs: pytest.fail(str(args)))
    restored._install_origami_analysis(load_analysis_session(path, live._analysis_record_types()))
    assert jobs == ['origami']
    assert restored.origami_last_rendered_plot_option == view
    assert restored.origami_result.origami_count == 12
    assert restored.origami_multi_template_overlays_building == set()
    assert restored.origami_template_result_view.get() == selected
    restored.origami_canvas.draw()


def test_roi_coarse_density_and_selected_detail_survive_load(tmp_path, monkeypatch):
    live = populate(app())
    # A single ROI retains its actual coarse map rather than the tiled placeholder.
    picks = live.origami_multi_template_results['A']['picks']
    picks.density_image = np.array([[1., 3.], [2., 4.]])
    picks.density_contrast = picks.density_image / 4
    picks.density_component_labels = np.ones((2, 2), dtype=int)
    picks.density_extent_nm = (-10., 10., -10., 10.)
    live.origami_template_result_view.set('A')
    live.origami_plot_option.set('Origami type counts')
    live._on_origami_template_result_selection()
    live.origami_plot_option.set('Coarse identification density')
    live.render_origami_plot()
    coarse = snapshot(live)
    live.origami_plot_option.set('Aligned density')
    live.render_origami_plot()
    live.origami_selected_index = 4
    live.origami_plot_option.set('Selected origami detail')
    live.render_origami_plot()
    detail = snapshot(live)
    path = tmp_path / 'detail.paintanalysis'
    save_analysis_session(path, live._capture_origami_analysis())
    restored = app()
    restored._run_worker = lambda fn: pytest.fail('Rebuilt a saved overlay')
    restored._install_origami_analysis(load_analysis_session(path, live._analysis_record_types()))
    assert restored.origami_selected_index == 4
    assert restored.origami_last_rendered_plot_option == 'Selected origami detail'
    for original, reopened in zip(detail, snapshot(restored)):
        if isinstance(original, np.ndarray): np.testing.assert_allclose(original, reopened)
        else: assert original == reopened
    restored.origami_plot_option.set('Coarse identification density')
    restored.render_origami_plot()
    for original, reopened in zip(coarse, snapshot(restored)):
        if isinstance(original, np.ndarray): np.testing.assert_allclose(original, reopened)
        else: assert original == reopened


def test_refinement_retains_classification_and_current_session_identity():
    live = populate(app())
    live.origami_template_result_view.set('A')
    live.origami_plot_option.set('Aligned density')
    live.render_origami_plot()
    live.origami_identification_generation = 100
    worker = Mock(return_value=('origami', {}))
    live._overlay_origami_worker = worker
    live._run_worker = lambda fn: fn()
    live.refine_origami_overlay_with_g5m()
    params = worker.call_args.args[3]
    assert params['template_name'] == 'A'
    assert params['identification_generation'] == 100
    assert params['preserve_pose']


def test_empty_loaded_classification_shows_empty_view_without_worker(tmp_path, monkeypatch):
    from dataclasses import replace
    live = populate(app())
    original = live.origami_multi_template_results['A']
    live.origami_multi_template_results['Empty'] = {
        'picks': replace(original['picks'], accepted_mask=np.zeros(36, dtype=bool)),
        'params': original['params']}
    live.origami_multi_template_counts['Empty'] = 0
    live.origami_template_result_view.set('Empty')
    live.origami_plot_option.set('Aligned density')
    path = tmp_path / 'empty.paintanalysis'
    save_analysis_session(path, live._capture_origami_analysis())
    restored = app()
    restored._run_worker = lambda fn: pytest.fail('Started a worker for empty classification')
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *args, **kwargs: pytest.fail(str(args)))
    restored._install_origami_analysis(load_analysis_session(path, live._analysis_record_types()))
    for view in ('Aligned density', 'Individual origami gallery', 'Site occupancy'):
        restored.origami_plot_option.set(view)
        restored._on_origami_template_result_selection()
        assert restored.origami_last_rendered_plot_option == view
        assert 'no origami' in restored.status.get().lower()
        assert not restored.origami_multi_template_overlays_building


def test_stale_diagnostics_from_previous_run_are_not_saved():
    live = populate(app())
    live.origami_review_cache = (('old-run',), {})
    live.origami_orientation_cache = (('old-run',), {'outdated': True})
    assert live._capture_origami_analysis()['diagnostics'] == {}


def test_load_invalidates_pending_display_updates():
    live = populate(app())
    live.origami_plot_option.set('Origami type counts')
    restored = app()
    restored.origami_zoom_render_request_id = 41
    restored.origami_zoom_render_after_id = 'old-zoom'
    restored.origami_footprint_refresh_after_id = 'old-footprints'
    restored.origami_zoom_render_running = True
    restored.origami_zoom_render_pending = True
    restored.origami_source_draw_signature = ('old-session',)
    restored.after_cancel = Mock()
    restored._install_origami_analysis(live._capture_origami_analysis())
    assert restored.after_cancel.call_args_list == [call('old-zoom'), call('old-footprints')]
    assert restored.origami_zoom_render_request_id == 42
    assert restored.origami_zoom_render_after_id is None
    assert restored.origami_footprint_refresh_after_id is None
    assert not restored.origami_zoom_render_running
    assert not restored.origami_zoom_render_pending
    assert restored.origami_source_draw_signature is None


def test_open_closed_pairs_survive_saved_session(tmp_path):
    live = populate(app())
    live.origami_multi_template_results = {
        'bit1_open': live.origami_multi_template_results['A'],
        'bit1_off': live.origami_multi_template_results['Full'],
    }
    live.origami_multi_template_counts = {'bit1_open': 12, 'bit1_off': 12}
    live.origami_template_result_view.set('All templates')
    live.origami_plot_option.set('Open vs closed')
    live.render_origami_plot()
    path = tmp_path / 'open_closed.paintanalysis'
    save_analysis_session(path, live._capture_origami_analysis())
    restored = app()
    restored._run_worker = lambda fn: pytest.fail('Open vs closed should not build overlays')
    restored._install_origami_analysis(load_analysis_session(path, live._analysis_record_types()))
    assert restored.origami_plot_option.get() == 'Open vs closed'
    assert restored.origami_last_rendered_plot_option == 'Open vs closed'
    assert [bar.get_height() for bar in restored.origami_figure.axes[0].patches] == [50, 50]
    assert [t.get_text() for t in restored.origami_figure.axes[0].get_xticklabels()] == ['bit1']


def test_all_templates_density_builds_sequentially_and_restores(tmp_path):
    from dataclasses import replace
    live = populate(app())
    original = live.origami_multi_template_results['A']
    live.origami_multi_template_results['Empty'] = {
        'picks': replace(original['picks'], accepted_mask=np.zeros(36, dtype=bool)),
        'params': original['params']}
    live.origami_multi_template_counts['Empty'] = 0
    live.origami_type_count_order = ['Full', 'A', 'Empty']
    live.origami_template_result_view.set('All templates')
    live.origami_plot_option.set('Aligned density')
    live.origami_show_theoretical_overlay.set(True)
    live.origami_show_site_diagnostics.set(True)
    queued = []
    live._run_worker = queued.append
    live._on_origami_template_result_selection()
    assert len(queued) == 1
    assert live.origami_multi_template_overlays_building == {'Full'}
    live.render_origami_plot()
    assert len(queued) == 1  # redraws do not duplicate a running worker
    for name in ('Full', 'A'):
        kind, payload = queued.pop(0)()
        assert kind == 'origami'
        assert payload['template_name'] == name
        live._plot_origami_analysis(payload)
    assert not queued
    assert not live.origami_multi_template_overlays_building
    assert live.origami_template_result_view.get() == 'All templates'
    assert live.origami_last_rendered_plot_option == 'Aligned density'
    panels = [axis for axis in live.origami_figure.axes if axis.get_xlabel() == 'aligned x (nm)']
    assert len(panels) == 2
    assert panels[0].get_title().startswith('Full\n12 origami')
    assert panels[1].get_title().startswith('A\n12 origami')
    for axis, name in zip(panels, ('Full', 'A')):
        cached = live.origami_multi_template_overlay_results[name]
        expected = gui.render_aligned_origami_density(
            cached['result'].aligned_points, **cached['render_settings'],
            symmetrize_180=cached['result'].symmetrized_180)
        np.testing.assert_allclose(axis.images[0].get_array(), expected['image'])
    assert all(axis.get_legend().get_visible() for axis in panels)
    live.origami_show_legends.set(False)
    live._toggle_origami_legends()
    assert all(not axis.get_legend().get_visible() for axis in panels)
    live.origami_show_legends.set(True)
    live._toggle_origami_legends()
    assert all(axis.get_legend().get_visible() for axis in panels)
    live.origami_show_legends.set(False)
    live.render_origami_plot()
    before = snapshot(live)
    path = tmp_path / 'all_density.paintanalysis'
    save_analysis_session(path, live._capture_origami_analysis())
    restored = app()
    restored._run_worker = lambda fn: pytest.fail('Rebuilt a saved density overlay')
    restored._install_origami_analysis(load_analysis_session(path, live._analysis_record_types()))
    assert restored.origami_template_result_view.get() == 'All templates'
    assert restored.origami_plot_option.get() == 'Aligned density'
    assert not restored.origami_show_legends.get()
    restored_panels = [axis for axis in restored.origami_figure.axes if axis.get_xlabel() == 'aligned x (nm)']
    assert all(not axis.get_legend().get_visible() for axis in restored_panels)
    after = snapshot(restored)
    assert len(before) == len(after)
    for original, reopened in zip(before, after):
        if isinstance(original, np.ndarray): np.testing.assert_allclose(original, reopened)
        else: assert original == reopened


def test_all_template_density_completion_does_not_switch_away_from_selected_view():
    live = populate(app())
    live.origami_template_result_view.set('All templates')
    live.origami_plot_option.set('Aligned density')
    queued = []
    live._run_worker = queued.append
    live.render_origami_plot()
    live.origami_plot_option.set('Origami type counts')
    live.render_origami_plot()
    _, payload = queued.pop(0)()
    live._plot_origami_analysis(payload)
    assert live.origami_last_rendered_plot_option == 'Origami type counts'
    assert not queued
    live.origami_plot_option.set('Aligned density')
    live.render_origami_plot()
    assert len(queued) == 1
