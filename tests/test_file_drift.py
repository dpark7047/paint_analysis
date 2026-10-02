import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paint_analysis_gui import apply_drift_correction, read_drift_csv


class FileDriftTests(unittest.TestCase):
    def test_applies_nm_drift_to_each_localization_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text(
                "Frame,x-drift (nm),y-drift (nm),z-drift (nm)\n"
                "2,260,-130,65\n"
                "0,0,0,0\n"
                "1,130,260,130\n",
                encoding="utf-8",
            )
            locs = pd.DataFrame(
                {
                    "frame": np.asarray([2, 0, 1, 2], dtype=np.uint32),
                    "x": np.asarray([10, 10, 10, 20], dtype=np.float32),
                    "y": np.asarray([5, 5, 5, 10], dtype=np.float32),
                    "z": np.asarray([3, 3, 3, 6], dtype=np.float32),
                }
            )
            info = [{"Frames": 3, "Width": 30, "Height": 30, "Pixelsize": 130.0}]

            corrected, drift, label = apply_drift_correction(
                locs,
                info,
                "file",
                1000,
                20.0,
                60.0,
                drift_file_path=path,
            )

        np.testing.assert_allclose(corrected["x"], [8, 10, 9, 18])
        np.testing.assert_allclose(corrected["y"], [6, 5, 3, 11])
        np.testing.assert_allclose(corrected["z"], [2.5, 3, 2, 5.5])
        np.testing.assert_allclose(drift["x"], [0, 1, 2])
        self.assertEqual(corrected["x"].dtype, np.float32)
        self.assertIn("drift.csv", label)
        np.testing.assert_allclose(locs["x"], [10, 10, 10, 20])

    def test_rejects_missing_drift_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text("Frame,x,y\n0,0,0\n2,1,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing 1 frame.*1"):
                read_drift_csv(path, frame_count=3, pixel_size_nm=130.0)

    def test_rejects_duplicate_drift_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text("Frame,x,y\n0,0,0\n0,1,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate frame"):
                read_drift_csv(path, frame_count=1, pixel_size_nm=130.0)


if __name__ == "__main__":
    unittest.main()


def test_exported_crop_drift_applies_to_full_field(tmp_path):
    from paint_analysis_gui import export_drift_csv, apply_drift_file
    # The crop only has detections in frames 1 and 3, but the estimated trace
    # covers the original acquisition, including empty frames.
    trace = pd.DataFrame({'x': [.25, 1., -.5, 2.], 'y': [-.5, 0., .5, 1.],
                          'z': [0., .1, .2, .3]})
    full = pd.DataFrame({'frame': [0, 1, 2, 3, 3], 'x': [100., 101., 99., 104., 200.],
                         'y': [40., 42., 45., 47., 80.], 'z': [1., 2., 3., 4., 5.]})
    info = [{'Frames': 4, 'Pixelsize': 130.}]
    destination = tmp_path / 'crop_drift.csv'
    assert export_drift_csv(trace, destination, 4, 130.) == 4
    exported = pd.read_csv(destination)
    assert list(exported) == ['Frame', 'x-drift (nm)', 'y-drift (nm)', 'z-drift (nm)']
    np.testing.assert_allclose(exported['x-drift (nm)'], trace['x'] * 130.)
    corrected, imported = apply_drift_file(full, info, destination)
    for axis in ('x', 'y', 'z'):
        np.testing.assert_allclose(imported[axis], trace[axis])
        np.testing.assert_allclose(corrected[axis], full[axis] - trace[axis].to_numpy()[full['frame']])
    # Unit-labelled export also converts correctly for a different coordinate pixel size.
    imported_other_scale = read_drift_csv(destination, 4, 65.)
    np.testing.assert_allclose(imported_other_scale['x'], 2 * trace['x'])


def test_drift_export_preserves_frame_index_and_existing_file_on_failure(tmp_path):
    import pytest
    from paint_analysis_gui import export_drift_csv
    destination = tmp_path / 'drift.csv'
    trace = pd.DataFrame({'x': [2., 0., 1.], 'y': [-2., 0., -1.]}, index=[2, 0, 1])
    export_drift_csv(trace, destination, 3, 100.)
    assert list(pd.read_csv(destination)['Frame']) == [2, 0, 1]
    np.testing.assert_allclose(read_drift_csv(destination, 3, 100.)['x'], [0, 1, 2])
    before = destination.read_bytes()
    with pytest.raises(ValueError, match='missing'):
        export_drift_csv(trace, destination, 4, 100.)
    assert destination.read_bytes() == before
    trace.loc[1, 'x'] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        export_drift_csv(trace, destination, 3, 100.)
    assert destination.read_bytes() == before
    assert list(tmp_path.iterdir()) == [destination]
