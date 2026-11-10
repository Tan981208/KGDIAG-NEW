"""Synthetic preprocessing and window-boundary checks; no external data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from kgdiag.data import (
    Preprocessor, Windows, continuity, numeric_values, read_frame, read_json,
    recording_slices,
)


def specification():
    return {"nodes": [
        {"name": "measurement", "kind": "continuous"},
        {"name": "constant", "kind": "continuous"},
        {"name": "state", "kind": "categorical"},
    ]}


def normal_frame():
    return pd.DataFrame({"measurement": [2.0, 4.0, 6.0, 8.0],
                         "constant": [5.0] * 4, "state": ["01", "02", "01", "02"]})


class PreprocessingTests(unittest.TestCase):
    def test_scaler_and_vocabulary_are_fit_only(self):
        p = Preprocessor.fit(normal_frame(), specification())
        frozen_state = copy.deepcopy(p.state)
        validation = pd.DataFrame({"measurement": [-1000.0, 2000.0],
                                   "constant": [8.0, 5.0], "state": ["99", "01"]})
        arrays = p.transform(validation)
        self.assertEqual(p.state, frozen_state)
        self.assertEqual(p.state["minimum"], [2.0, 5.0, 0.0])
        self.assertEqual(p.state["span"], [6.0, 1.0, 1.0])
        self.assertEqual(p.state["categories"], {"2": ["01", "02"]})
        np.testing.assert_allclose(arrays["target"][:, 0], [-167.0, 333.0])
        # Transformation retains out-of-Fit-range values rather than clipping them.
        self.assertLess(arrays["target"][0, 0], 0)
        self.assertGreater(arrays["target"][1, 0], 1)

    def test_zero_fit_span_uses_unit_scale(self):
        p = Preprocessor.fit(normal_frame(), specification())
        base = p.transform(normal_frame())
        np.testing.assert_array_equal(base["target"][:, 1], np.zeros(4))
        changed = normal_frame()
        changed.loc[0, "constant"] = 7.0
        self.assertEqual(float(p.transform(changed)["target"][0, 1]), 2.0)
        self.assertEqual(p.state["span"][1], 1.0)

    def test_unseen_and_missing_categories_have_zero_encoding(self):
        p = Preprocessor.fit(normal_frame(), specification())
        queries = normal_frame().copy()
        queries["state"] = ["unseen", None, "01", "02"]
        arrays = p.transform(queries)
        np.testing.assert_array_equal(arrays["categories"][:, 2], [0, 0, 1, 2])
        np.testing.assert_array_equal(arrays["temporal"][:, 2:],
                                      [[0, 0], [0, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(arrays["nodes"][:, 2], np.zeros(4))
        self.assertEqual(arrays["target"].shape, (4, 2))

    def test_numeric_node_order_survives_sorted_json_keys(self):
        nodes = [{"name": f"x{i}", "kind": "categorical" if i in (2, 10) else "continuous"}
                 for i in range(12)]
        frame = pd.DataFrame({f"x{i}": [float(i), float(i + 1), float(i + 2)]
                              for i in range(12)})
        frame["x2"] = ["B", "A", "B"]
        frame["x10"] = ["Z", "X", "Y"]
        fitted = Preprocessor.fit(frame, {"nodes": nodes})
        state = json.loads(json.dumps(fitted.state, sort_keys=True))
        self.assertEqual(list(state["categories"]), ["10", "2"])
        restored = Preprocessor(state)
        expected = fitted.transform(frame)
        actual = restored.transform(frame)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual["temporal"][0, 10:], [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(actual["categories"][:, 2], [2, 1, 2])
        np.testing.assert_array_equal(actual["categories"][:, 10], [3, 1, 2])

    def test_csv_preserves_categorical_codes_and_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            normal_frame().to_csv(path, index=False)
            loaded = read_frame(path, specification())
            self.assertEqual(loaded["state"].tolist(), ["01", "02", "01", "02"])
            p = Preprocessor.fit(loaded, specification())
            state_path = Path(tmp) / "preprocessing.json"
            state_path.write_text(json.dumps(p.state, sort_keys=True), encoding="utf-8")
            restored = Preprocessor(read_json(state_path))
            np.testing.assert_array_equal(restored.transform(loaded)["temporal"],
                                          p.transform(loaded)["temporal"])

    def test_numeric_graph_input_excludes_categories(self):
        frame = normal_frame()
        frame.loc[0, "measurement"] = np.nan
        raw = numeric_values(frame, specification())
        self.assertTrue(np.isnan(raw[:, 2]).all())
        self.assertTrue(np.isnan(raw[0, 0]))
        np.testing.assert_array_equal(raw[1:, 0], [4.0, 6.0, 8.0])


class WindowTests(unittest.TestCase):
    def make_recordings(self):
        spec = specification()
        spec.update(segment_column="recording", timestamp_column="time", max_gap_seconds=1.5)
        seconds = [0, 1, 2, 3, 4, 5, 0, 1, 2, 10, 11, 12]
        frame = pd.DataFrame({"measurement": np.arange(12, dtype=float),
                              "constant": [5.0] * 12, "state": ["01"] * 12,
                              "recording": ["first"] * 6 + ["second"] * 6,
                              "time": [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(seconds=s)
                                       for s in seconds]})
        p = Preprocessor.fit(frame.iloc[:6], spec)
        return spec, frame, p.transform(frame)

    def test_recording_and_time_gap_boundaries(self):
        spec, frame, arrays = self.make_recordings()
        adjacent = continuity(frame, spec)
        np.testing.assert_array_equal(np.flatnonzero(~adjacent), [0, 6, 9])
        slices = recording_slices(frame, spec)
        self.assertEqual([(s.start, s.stop) for s in slices], [(0, 6), (6, 9), (9, 12)])
        windows = Windows(arrays, adjacent, window=3)
        np.testing.assert_array_equal(windows.endpoints, [2, 3, 4, 5, 8, 11])
        self.assertEqual(len(windows), 6)
        for i, endpoint in enumerate(windows.endpoints):
            tensors = windows[i]
            for k, tensor in zip(("temporal", "nodes", "categories", "target"), tensors):
                np.testing.assert_array_equal(tensor.numpy(), arrays[k][endpoint - 2:endpoint + 1])

    def test_nonincreasing_time_also_breaks_windows(self):
        spec, frame, arrays = self.make_recordings()
        frame.loc[4, "time"] = frame.loc[3, "time"]
        frame.loc[5, "time"] = frame.loc[2, "time"]
        adjacent = continuity(frame, spec)
        self.assertFalse(adjacent[4])
        self.assertFalse(adjacent[5])
        windows = Windows(arrays, adjacent, window=2)
        self.assertNotIn(4, windows.endpoints)
        self.assertNotIn(5, windows.endpoints)

    def test_invalid_policy_error_or_skip_never_imputes(self):
        spec, frame, _ = self.make_recordings()
        p = Preprocessor.fit(frame.iloc[:6], spec)
        frame.loc[4, "measurement"] = np.nan
        arrays = p.transform(frame)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            Windows(arrays, continuity(frame, spec), window=3, invalid_policy="error")
        windows = Windows(arrays, continuity(frame, spec), window=3, invalid_policy="skip")
        np.testing.assert_array_equal(windows.invalid_rows, [4])
        np.testing.assert_array_equal(windows.endpoints, [2, 3, 8, 11])
        self.assertTrue(np.isnan(arrays["nodes"][4, 0]))
        for endpoint in windows.endpoints:
            self.assertFalse(endpoint - 2 <= 4 <= endpoint)

    def test_missing_recording_id_is_rejected(self):
        spec, frame, _ = self.make_recordings()
        frame.loc[4, "recording"] = None
        with self.assertRaisesRegex(ValueError, "recording"):
            continuity(frame, spec)

    def test_no_valid_window_is_explicit(self):
        spec, frame, arrays = self.make_recordings()
        with self.assertRaisesRegex(ValueError, "No valid windows"):
            Windows(arrays, continuity(frame, spec), window=7)


if __name__ == "__main__":
    unittest.main()
