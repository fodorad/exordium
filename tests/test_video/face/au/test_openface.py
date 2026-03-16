"""Tests for exordium.video.face.au.openface module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from exordium.video.face.au import read_openface_au


def _write_openface_csv(path: Path, rows: list[dict]) -> None:
    """Helper: write a minimal OpenFace-style CSV."""
    au_r_cols = [
        f"AU{i:02d}_r" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    ]
    au_c_cols = [
        f"AU{i:02d}_c" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
    ]
    # Total AU columns: 17 r + 18 c = 35
    au_cols = au_r_cols + au_c_cols
    assert len(au_cols) == 35, f"Expected 35 AU columns, got {len(au_cols)}"
    cols = ["frame", "face_id", "timestamp", "confidence", "success"] + au_cols
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


class TestReadOpenfaceAu(unittest.TestCase):
    """Tests for read_openface_au."""

    def _make_row(self, frame_id=0, face_id=0, confidence=0.9):
        au_r_vals = [0.5] * 17
        au_c_vals = [1.0] * 18
        return [frame_id, face_id, 0.033, confidence, 1] + au_r_vals + au_c_vals

    def test_basic_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "openface.csv"
            rows = [self._make_row(i) for i in range(5)]
            _write_openface_csv(csv_path, rows)
            frame_ids, au_values, au_names = read_openface_au(csv_path)
            self.assertEqual(au_values.ndim, 2)
            self.assertEqual(au_values.shape[1], 35)
            self.assertEqual(len(frame_ids), 5)

    def test_low_confidence_filtered_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "openface.csv"
            rows = [
                self._make_row(0, confidence=0.5),  # below threshold
                self._make_row(1, confidence=0.9),  # above threshold
                self._make_row(2, confidence=0.9),  # above threshold
            ]
            _write_openface_csv(csv_path, rows)
            frame_ids, au_values, _ = read_openface_au(csv_path, confidence_thr=0.85)
            self.assertEqual(len(frame_ids), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_openface_au("/nonexistent/path/openface.csv")

    def test_all_low_confidence_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "openface.csv"
            rows = [self._make_row(i, confidence=0.1) for i in range(3)]
            _write_openface_csv(csv_path, rows)
            with self.assertRaises(ValueError):
                read_openface_au(csv_path, confidence_thr=0.85)

    def test_returns_au_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "openface.csv"
            rows = [self._make_row(i) for i in range(3)]
            _write_openface_csv(csv_path, rows)
            _, _, au_names = read_openface_au(csv_path)
            self.assertIsInstance(au_names, np.ndarray)
            self.assertEqual(len(au_names), 35)

    def test_multiple_faces_selects_biggest(self):
        """When multiple faces are present, the biggest by bounding box is selected."""
        au_r_cols = [
            f"AU{i:02d}_r" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        ]
        au_c_cols = [
            f"AU{i:02d}_c"
            for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
        ]
        au_cols = au_r_cols + au_c_cols
        x_cols = [f"x_{i}" for i in range(68)]
        y_cols = [f"y_{i}" for i in range(68)]
        cols = (
            ["frame", "face_id", "timestamp", "confidence", "success"] + x_cols + y_cols + au_cols
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "multi_face.csv"

            # Face 0: small bounding box, Face 1: large bounding box
            rows = []
            for face_id in [0, 1]:
                x_vals = list(range(68)) if face_id == 0 else list(range(0, 680, 10))
                y_vals = list(range(68)) if face_id == 0 else list(range(0, 680, 10))
                au_vals = [0.5] * 17 + [1.0] * 18
                row = [0, face_id, 0.033, 0.95, 1] + x_vals + y_vals + au_vals
                rows.append(row)

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(csv_path, index=False)

            frame_ids, au_values, _ = read_openface_au(csv_path)
            self.assertEqual(len(frame_ids), 1)


if __name__ == "__main__":
    unittest.main()
