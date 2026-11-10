"""Fit-only input encoding with explicit recording and window boundaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_frame(path, specification):
    # Read categories as strings so state codes do not acquire an ordinal metric.
    categorical = {n["name"]: "string" for n in specification["nodes"] if n["kind"] == "categorical"}
    frame = pd.read_csv(path, dtype=categorical)
    expected = [n["name"] for n in specification["nodes"]]
    if len(expected) != len(set(expected)) or not expected:
        raise ValueError("Node names must be nonempty and unique")
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    return frame


def numeric_values(frame, specification):
    out = np.full((len(frame), len(specification["nodes"])), np.nan, dtype=np.float64)
    for i, node in enumerate(specification["nodes"]):
        if node["kind"] == "continuous":
            out[:, i] = pd.to_numeric(frame[node["name"]], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        elif node["kind"] != "categorical":
            raise ValueError(f"Unknown kind for {node['name']}")
    return out


def continuity(frame, specification):
    """adjacent[t] means t-1 and t belong to one usable recording timeline."""
    adjacent = np.ones(len(frame), dtype=bool)
    if not len(frame):
        return adjacent
    adjacent[0] = False
    segment = specification.get("segment_column")
    if segment:
        if segment not in frame or frame[segment].isna().any():
            raise ValueError("Every row needs a recording/segment identifier")
        values = frame[segment].astype(str).to_numpy()
        adjacent[1:] &= values[1:] == values[:-1]
    timestamp = specification.get("timestamp_column")
    if timestamp:
        if timestamp not in frame:
            raise ValueError("Missing timestamp column")
        times = pd.to_datetime(frame[timestamp], errors="raise", utc=True)
        if times.isna().any():
            raise ValueError("Missing timestamp")
        difference = np.diff(times.astype("int64").to_numpy()) / 1e9
        adjacent[1:] &= difference > 0
        maximum = specification.get("max_gap_seconds")
        if maximum is not None:
            if maximum <= 0:
                raise ValueError("max_gap_seconds must be positive")
            adjacent[1:] &= difference <= maximum
    return adjacent


def recording_slices(frame, specification):
    starts = np.flatnonzero(~continuity(frame, specification))
    return [slice(int(a), int(b)) for a, b in zip(starts, np.r_[starts[1:], len(frame)])]


class Preprocessor:
    def __init__(self, state):
        self.state = state
        self.schema = state["schema"]

    @classmethod
    def fit(cls, frame, specification):
        nodes = specification["nodes"]
        raw = numeric_values(frame, specification)
        continuous = [i for i, n in enumerate(nodes) if n["kind"] == "continuous"]
        if not continuous or not len(frame):
            raise ValueError("Fit requires continuous targets and observations")
        low, span = np.zeros(len(nodes)), np.ones(len(nodes))
        categories, sizes = {}, [0] * len(nodes)
        for i, node in enumerate(nodes):
            if i in continuous:
                valid = raw[np.isfinite(raw[:, i]), i]
                if not len(valid):
                    raise ValueError(f"No valid Fit observations: {node['name']}")
                low[i] = valid.min()
                extent = valid.max() - low[i]
                span[i] = extent if extent > 0 else 1.0
            else:
                known = sorted(frame[node["name"]].dropna().astype(str).unique().tolist())
                if not known:
                    raise ValueError(f"No known Fit states: {node['name']}")
                categories[str(i)] = known
                sizes[i] = len(known)
        schema = dict(nodes=nodes, names=[n["name"] for n in nodes], continuous_indices=continuous,
                      categorical_cardinalities=sizes)
        for key, field in [("type", "signal_type"), ("stage", "stage"), ("family", "family")]:
            names = [str(n.get(field) or "UNKNOWN") for n in nodes]
            vocabulary = {v: i for i, v in enumerate(sorted(set(names) | {"UNKNOWN"}))}
            schema[f"node_{key}_ids"] = [vocabulary[v] for v in names]
        state = dict(specification=specification, schema=schema, minimum=low.tolist(),
                     span=span.tolist(), categories=categories)
        return cls(state)

    def transform(self, frame):
        raw = numeric_values(frame, self.state["specification"])
        indices = self.schema["continuous_indices"]
        values = np.zeros_like(raw)
        values[:, indices] = ((raw - np.asarray(self.state["minimum"])) /
                               np.asarray(self.state["span"]))[:, indices]
        ids = np.zeros(raw.shape, dtype=np.int64)
        one_hot = []
        for key in sorted(self.state["categories"], key=int):
            vocabulary = self.state["categories"][key]
            i = int(key)
            mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
            codes = frame[self.schema["names"][i]].astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
            ids[:, i] = codes
            one_hot.append((codes[:, None] == np.arange(1, len(vocabulary) + 1)[None, :]).astype(np.float32))
        temporal = np.concatenate([values[:, indices]] + one_hot, axis=1).astype(np.float32)
        return dict(temporal=temporal, nodes=values.astype(np.float32), categories=ids,
                    target=values[:, indices].astype(np.float32))


class Windows(Dataset):
    """Windows never cross a split, recording boundary, nonmonotone time, or gap."""
    def __init__(self, arrays, adjacent, window, stride=1, invalid_policy="error"):
        if window < 1 or stride < 1 or invalid_policy not in ("error", "skip"):
            raise ValueError("Invalid window/stride/invalid policy")
        self.arrays = arrays
        self.window = window
        finite = np.isfinite(arrays["temporal"]).all(1) & np.isfinite(arrays["nodes"]).all(1)
        self.invalid_rows = np.flatnonzero(~finite)
        if len(self.invalid_rows) and invalid_policy == "error":
            raise ValueError("Nonfinite continuous input: fix declared validity handling or explicitly select invalid_policy=skip")
        n = len(finite)
        endpoints = np.arange(window - 1, n, stride)
        breaks = np.r_[0, np.cumsum(~adjacent)]
        invalid = np.r_[0, np.cumsum(~finite)]
        start = endpoints - window + 1
        usable = ((breaks[endpoints + 1] - breaks[start + 1]) == 0) & ((invalid[endpoints + 1] - invalid[start]) == 0)
        self.endpoints = endpoints[usable]
        if not len(self.endpoints):
            raise ValueError("No valid windows in this split")

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        rows = slice(end - self.window + 1, end + 1)
        return tuple(torch.from_numpy(self.arrays[k][rows]) for k in ("temporal", "nodes", "categories", "target"))
