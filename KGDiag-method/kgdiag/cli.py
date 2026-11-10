"""Train and apply KGDiag using explicitly supplied normal splits and graph data."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .association import components, loss_terms
from .data import Preprocessor, Windows, continuity, numeric_values, read_frame, read_json, recording_slices
from .evaluation import evaluate
from .graph import build_graph
from .model import KGDiag
from .scoring import combine, fit_pot, fit_standardization, residual_rarity
from .training import capped_backward


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_checkpoint(path, device="cpu"):
    options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _windows(frame, preprocessor, config):
    return Windows(preprocessor.transform(frame), continuity(frame, preprocessor.state["specification"]),
                   config["model"]["window"], config["data"]["stride"], config["data"]["invalid_policy"])


def _device_batch(batch, device):
    return tuple(value.to(device) for value in batch)


@torch.no_grad()
def validation_mse(model, dataset, batch_size, device):
    model.eval()
    total, count = 0., 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        tx, nodes, cats, target = _device_batch(batch, device)
        error = (model(tx, nodes, cats)["reconstruction"] - target).double().square()
        total += float(error.sum())
        count += error.numel()
    return total / count


@torch.no_grad()
def infer_components(model, dataset, batch_size, device, evidence=False):
    model.eval()
    keys = ["R", "DT", "DG"] + (["residual", "node_graph"] if evidence else [])
    arrays = {name: [] for name in keys}
    if evidence:
        arrays["relation"] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        tx, nodes, cats, target = _device_batch(batch, device)
        output = model(tx, nodes, cats)
        values = components(output, target, output["support"])
        for name in keys:
            arrays[name].append(values[name].cpu().numpy())
        if evidence:
            parts = []
            for relation in range(len(model.relation_names)):
                mask = model.graph_support & (model.graph_relation_ids == relation)
                parts.append(values["edge"][:, mask].sum(-1))
            arrays["relation"].append(torch.stack(parts, dim=-1).cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in arrays.items()}


def train(args):
    cfg, spec = read_json(args.config), read_json(args.schema)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; existing artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    _seed(cfg["training"]["seed"])
    fit, validation = read_frame(args.fit, spec), read_frame(args.validation, spec)
    sources = dict(fit=args.fit, validation=args.validation)
    if args.calibration:
        sources["calibration"] = args.calibration
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Fit, Validation, and Calibration must be distinct explicit inputs")
    pre = Preprocessor.fit(fit, spec)
    if spec.get("timestamp_column"):
        seen = set()
        for path in sources.values():
            f = read_frame(path, spec)
            timestamps = pd.to_datetime(f[spec["timestamp_column"]], utc=True).astype("int64")
            segment = f[spec["segment_column"]].astype(str) if spec.get("segment_column") else [""] * len(f)
            keys = set(zip(segment, timestamps))
            if seen & keys:
                raise ValueError("Normal splits overlap in recording/timestamp identity")
            seen |= keys
    edges = read_json(args.edges) if args.edges else []
    # Segment stability is checked within recordings, never on concatenated seams.
    segments = []
    for record in recording_slices(fit, spec):
        for indices in np.array_split(np.arange(record.start, record.stop), cfg["graph"]["fit_segment_count"]):
            if len(indices):
                segments.append(slice(int(indices[0]), int(indices[-1]) + 1))
    graph = build_graph(pre.schema, numeric_values(fit, spec), numeric_values(validation, spec),
                        edges, cfg["graph"], fit_segments=segments)
    fit_ds, val_ds = _windows(fit, pre, cfg), _windows(validation, pre, cfg)
    model = KGDiag(cfg["model"], pre.schema, graph).to(args.device)
    training = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    loader = DataLoader(fit_ds, batch_size=training["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(training["seed"]))
    reference_end = None
    if spec.get("timestamp_column"):
        frames = [fit, validation]
        if args.calibration:
            frames.append(read_frame(args.calibration, spec))
        reference_end = max(pd.to_datetime(f[spec["timestamp_column"]], utc=True).max().isoformat() for f in frames)
    checkpoint = dict(config=cfg, preprocessing=pre.state, graph=_json_value(graph),
                      normal_source_sha256=hashes, reference_available_after=reference_end)
    write_json(output / "config.json", cfg)
    write_json(output / "graph.json", graph)
    write_json(output / "preprocessing.json", pre.state)
    best, history = float("inf"), []
    epoch = 0
    try:
        for epoch in range(1, training["epochs"] + 1):
            model.train()
            count, total, max_ratio = 0, 0., 0.
            for batch in loader:
                tx, nodes, cats, target = _device_batch(batch, args.device)
                parts = loss_terms(model(tx, nodes, cats), target)
                statistics = capped_backward(model, parts, training["lambda_t"], training["lambda_g"], training["gradient_cap"])
                optimizer.step()
                count += len(tx)
                total += statistics["reconstruction_loss"] * len(tx)
                max_ratio = max(max_ratio, statistics["effective_rho_after"])
            val = validation_mse(model, val_ds, training["batch_size"], args.device)
            row = dict(epoch=epoch, train_mse=total/count, validation_mse=val, max_shared_auxiliary_ratio=max_ratio)
            history.append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = val
                torch.save(dict(checkpoint, model=model.state_dict(), epoch=epoch, validation_mse=val), output / "best.pt")
            torch.save(dict(checkpoint, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                            torch_rng=torch.get_rng_state(), numpy_rng=_json_value(np.random.get_state()),
                            loader_rng=loader.generator.get_state(), history=history), output / "last.pt")
            write_json(output / "training.json", history)
    except KeyboardInterrupt:
        torch.save(dict(checkpoint, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        partial_epoch=True, torch_rng=torch.get_rng_state(), numpy_rng=_json_value(np.random.get_state())), output / "interrupted.pt")
        raise
    best_checkpoint = load_checkpoint(output / "best.pt", args.device)
    model.load_state_dict(best_checkpoint["model"])
    normal = infer_components(model, fit_ds, training["batch_size"], args.device)
    scaling = fit_standardization(normal)
    best_checkpoint["standardization"] = scaling
    readout = cfg["scoring"]["readout"]
    # Save the trained model even if a degenerate scale or POT fit stops calibration.
    torch.save(best_checkpoint, output / "best.pt")
    threshold = fit_pot(combine(normal, scaling, readout), **cfg["scoring"]["pot"])
    best_checkpoint.update(pot=threshold, readout=readout)
    if args.calibration:
        cal_ds = _windows(read_frame(args.calibration, spec), pre, cfg)
        cal = infer_components(model, cal_ds, training["batch_size"], args.device, evidence=True)
        best_checkpoint["normal_residual_reference"] = torch.from_numpy(cal["residual"])
    torch.save(best_checkpoint, output / "best.pt")
    write_json(output / "calibration.json", dict(standardization=scaling, pot=threshold, readout=readout))
    print("Saved calibrated best.pt", flush=True)


def infer(args):
    saved = load_checkpoint(args.checkpoint)
    if "pot" not in saved:
        raise ValueError("Checkpoint has no successful normal-score POT calibration")
    pre, cfg = Preprocessor(saved["preprocessing"]), saved["config"]
    spec = pre.state["specification"]
    frame = read_frame(args.data, spec)
    dataset = _windows(frame, pre, cfg)
    model = KGDiag(cfg["model"], pre.schema, saved["graph"]).to(args.device)
    model.load_state_dict(saved["model"])
    values = infer_components(model, dataset, cfg["training"]["batch_size"], args.device, args.evidence)
    score = combine(values, saved["standardization"], saved["readout"])
    result = pd.DataFrame(dict(row_index=dataset.endpoints, R=values["R"], DT=values["DT"], DG=values["DG"],
                               score=score, threshold=saved["pot"]["threshold"], alarm=score > saved["pot"]["threshold"]))
    result["adjacent"] = np.r_[False, np.diff(dataset.endpoints) == 1] & continuity(frame, spec)[dataset.endpoints]
    if spec.get("segment_column"):
        result["segment_id"] = frame[spec["segment_column"]].astype(str).iloc[dataset.endpoints].to_numpy()
    if spec.get("timestamp_column"):
        times = pd.to_datetime(frame[spec["timestamp_column"]], utc=True).iloc[dataset.endpoints]
        available = times >= pd.Timestamp(saved["reference_available_after"])
        if not available.all() and not args.retrospective:
            raise ValueError("Some query times precede the frozen model/reference availability; use --retrospective only for retrospective analysis")
        result["timestamp"] = times.astype(str).to_numpy()
        result["reference_available"] = available.to_numpy()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inference output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "scores.csv", index=False)
    if args.evidence:
        targets = pre.schema["continuous_indices"]
        eta = cfg["diagnosis"]["eta"]
        evidence = dict(row_index=dataset.endpoints, residual=values["residual"],
                        reconstruction_contribution=values["residual"] / len(targets),
                        node_graph=values["node_graph"], relation=values["relation"],
                        mixed_node=eta*values["residual"]+(1-eta)*values["node_graph"][:, targets],
                        informative_graph_rows=np.asarray(saved["graph"]["support"]).sum(-1)>1)
        if "normal_residual_reference" in saved:
            evidence["residual_rarity"] = residual_rarity(saved["normal_residual_reference"], values["residual"])
        np.savez_compressed(output / "evidence.npz", **evidence)
        write_json(output / "evidence_schema.json", dict(node_names=pre.schema["names"],
                   target_names=[pre.schema["names"][i] for i in targets], relation_names=saved["graph"]["relation_names"],
                   relation_available=[bool(np.any(np.asarray(saved["graph"]["relation_ids"])[np.asarray(saved["graph"]["support"], bool)] == i))
                                       for i in range(len(saved["graph"]["relation_names"]))], eta=eta))
    write_json(output / "input_audit.json", dict(rows=len(frame), scored_rows=len(dataset),
               invalid_row_indices=dataset.invalid_rows, window=cfg["model"]["window"], readout=saved["readout"]))


def metrics(args):
    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    if args.column not in labels:
        raise ValueError("Label column missing")
    indices = scores["row_index"].to_numpy(dtype=np.int64)
    if len(indices) == 0 or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(labels):
        raise ValueError("Invalid endpoint-to-label alignment")
    if not np.all(scores["threshold"] == scores["threshold"].iloc[0]):
        raise ValueError("Expected a single frozen POT threshold")
    adjacent = np.r_[False, np.diff(indices) == 1]
    if "adjacent" in scores:
        if scores["adjacent"].isna().any() or not scores["adjacent"].isin([True, False, 0, 1]).all():
            raise ValueError("Invalid saved adjacency values")
        adjacent &= scores["adjacent"].to_numpy(dtype=bool)
    if args.schema:
        specification = read_json(args.schema)
        adjacent &= continuity(labels, specification)[indices]
        if "timestamp" in scores and specification.get("timestamp_column"):
            expected = pd.to_datetime(labels[specification["timestamp_column"]], utc=True).iloc[indices].reset_index(drop=True)
            actual = pd.to_datetime(scores["timestamp"], utc=True)
            if not expected.equals(actual):
                raise ValueError("Timestamp mismatch between scores and labels")
        if "segment_id" in scores and specification.get("segment_column"):
            expected = labels[specification["segment_column"]].astype(str).iloc[indices].to_numpy()
            if not np.array_equal(expected, scores["segment_id"].astype(str).to_numpy()):
                raise ValueError("Recording identity mismatch between scores and labels")
    report = evaluate(labels[args.column].to_numpy()[indices], scores["score"].to_numpy(), scores["threshold"].iloc[0], adjacent)
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description="KGDiag method code: explicit normal splits, frozen score calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    for name in ("config", "schema", "fit", "validation", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--edges", help="JSON engineering edge list; omitted means statistical/self graph only")
    p.add_argument("--calibration", help="Separate normal reference for optional residual rarity")
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=train)
    p = sub.add_parser("infer")
    for name in ("checkpoint", "data", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--evidence", action="store_true")
    p.add_argument("--retrospective", action="store_true")
    p.set_defaults(run=infer)
    p = sub.add_parser("evaluate")
    for name in ("scores", "labels", "column", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--schema", help="Same registry for recording/timestamp boundary checks")
    p.set_defaults(run=metrics)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
