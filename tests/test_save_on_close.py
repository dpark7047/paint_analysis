from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from paint_analysis_gui import PaintAnalysisApp


@pytest.mark.parametrize('choice', [True, False, None])
def test_close_prompts_for_existing_analysis(choice):
    app = SimpleNamespace(origami_multi_template_results={'A': {}},
                          _origami_analysis_io=Mock(), destroy=Mock())
    with patch('paint_analysis_gui.messagebox.askyesnocancel', return_value=choice) as prompt, \
         patch('paint_analysis_gui.latest_development_session_request', return_value=None), \
         patch('paint_analysis_gui.set_next_startup_session'):
        PaintAnalysisApp._on_close(app)
    prompt.assert_called_once()
    if choice is True:
        app._origami_analysis_io.assert_called_once_with(loading=False, close_after_save=True)
    else:
        app._origami_analysis_io.assert_not_called()
    assert app.destroy.call_count == (1 if choice is False else 0)


@pytest.mark.parametrize('error', [None, 'disk full'])
def test_only_successful_save_closes_without_prompting_again(error):
    app = SimpleNamespace(origami_pick_result=object(), analysis_io_busy=True,
                          _close_after_analysis_save=True, _analysis_io_dialog=Mock(),
                          _remember_file_dialog_dir=Mock(), status=Mock(), destroy=Mock())
    with patch('paint_analysis_gui.messagebox.askyesnocancel') as prompt, \
         patch('paint_analysis_gui.messagebox.showerror') as showerror, \
         patch('paint_analysis_gui.latest_development_session_request', return_value=None), \
         patch('paint_analysis_gui.set_next_startup_session'):
        PaintAnalysisApp._complete_origami_analysis_io(app, Path('saved.paintanalysis'), False, None, error)
    prompt.assert_not_called()
    assert app.destroy.call_count == (0 if error else 1)
    assert showerror.call_count == (1 if error else 0)
    assert not app.analysis_io_busy
    assert '_close_after_analysis_save' not in vars(app)


def test_cancel_save_filename_keeps_app_open():
    app = SimpleNamespace(loaded=object(), origami_pick_result=object(),
                          origami_identification_running=False, session_load_in_progress=False,
                          _file_dialog_initial_dir=lambda: Path('/tmp'), destroy=Mock())
    with patch('paint_analysis_gui.filedialog.asksaveasfilename', return_value=''):
        PaintAnalysisApp._origami_analysis_io(app, loading=False, close_after_save=True)
    app.destroy.assert_not_called()
    assert not vars(app).get('_close_after_analysis_save')


def test_normal_save_does_not_close_app():
    app = SimpleNamespace(analysis_io_busy=True, _remember_file_dialog_dir=Mock(),
                          status=Mock(), destroy=Mock())
    PaintAnalysisApp._complete_origami_analysis_io(app, Path('saved.paintanalysis'), False, None, None)
    app.destroy.assert_not_called()
