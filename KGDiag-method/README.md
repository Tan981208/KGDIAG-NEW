# KGDiag method code

**Knowledge-Guided Anomaly Detection with Structured Diagnostic Evidence for Industrial Time Series**

This package implements KGDiag's temporal/spatial reconstruction model, typed
graph construction, constrained association training, anomaly scoring, and
variable/relation evidence. It contains source, configuration, tests, and data
access instructions. Dataset files, model weights, prediction caches, tables,
and figures are not bundled. See `DATASETS.md` for official dataset access.

## Install

Use Python 3.10 or later. Install a PyTorch build appropriate for the CPU/GPU
first, then install the remaining requirements:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m kgdiag --help
```

The development checks use Python 3.10, PyTorch 1.12, NumPy 1.24, pandas 2.1,
SciPy 1.14, and scikit-learn 1.5 on CPU. GPU execution is selected explicitly
with `--device cuda`; use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for a compatible build. Tests create tiny synthetic tensors and temporary files.
They require no dataset downloads, network access, or existing checkpoints.

## Method

1. Fit variable-wise MinMax statistics and categorical vocabularies on **normal
   Fit** only. Categorical state codes are not treated as continuous targets.
2. Build directed graph tensors from sourced engineering edges and normal-data
   Pearson auxiliary edges. Rows receive messages, columns send them.
3. Encode temporal input with width-three circular Conv1d, positional encoding,
   and temporal attention. A graph branch embeds each physical variable,
   constrains attention by support and type bias, and pools nodes at each time.
4. Both cross-attention directions read the original branch features. Residual
   LayerNorm and a sigmoid gate precede the decoder `d -> 4d -> 4d -> Nrec`,
   with ReLU after its first two linear layers.
5. Train with `2 * reconstruction_MSE` plus temporal and graph association
   minimax terms, each weighted by 0.001. Preserve stop-gradient arguments.
   The aggregate auxiliary gradient on shared parameters is capped at 10% of
   the reconstruction gradient norm. This is not an Adam update-size guarantee.
6. Score endpoints by `zR - zDT + zDG`. Normal Fit provides component means and
   population standard deviations and the static POT threshold. The `R`
   readout supports reconstruction-only scoring without discrepancy scaling.
7. Produce continuous-target mixed scores `0.9 * residual + 0.1 * node_graph`,
   reconstruction contributions, optional normal residual rarity, and additive
   relation-type contributions. These are inspection evidence, not causal labels.

Training averages discrepancies across layers. Detection sums layers and
multiplies by 10: DT averages all queries in the current window; DG uses the
endpoint graph. Diagnostic edge evidence averages layers and heads before node
or type summation. With N nodes and K graph layers, all type contributions sum
to `N * DG / (10 * K)`. Normalization is over the **complete supported
neighborhood**, never separately within each relation type. A singleton
neighborhood is marked uninformative for allocation changes.

`causal_lag` may be an evidence-supported edge type and may carry lag metadata.
This backbone does not read lag-shifted neighbor inputs. Discrete states enter
both branches as context; only continuous measurements are reconstructed.

## Prepare input files

Prepare separate CSV files for normal Fit, normal Validation, optional normal
Calibration, and the observations to score. Split complete recordings before
fitting any statistics. Preserve timestamps and recording identifiers. No
routine in training or score calibration accepts test labels.

The CSV header must contain every node in a user-supplied JSON registry, e.g.:

```json
{
  "timestamp_column": "timestamp",
  "segment_column": "recording",
  "max_gap_seconds": 1.5,
  "nodes": [
    {"name": "flow", "kind": "continuous", "signal_type": "sensor", "stage": "stage1", "family": "flow"},
    {"name": "level", "kind": "continuous", "signal_type": "sensor", "stage": "stage1", "family": "level"},
    {"name": "pump_state", "kind": "categorical", "signal_type": "state", "stage": "stage1", "family": "pump"}
  ]
}
```

This is an interface example, not an asserted engineering graph. Use only
metadata supported by your dataset; absent metadata become `UNKNOWN`. Omit
timestamp/recording fields only for genuinely contiguous arrays whose order
and boundaries have already been established. Without timestamps, the runner
cannot certify calendar-time availability or disjointness of normal inputs.

The selected node registry determines retained variables. No column is removed
based on Test behavior. Constant Fit channels use unit scale and remain
observable; constants and categorical codes cannot generate Pearson links.
Unknown/missing states receive category ID 0 and zero categorical-value vectors.
Finite out-of-range continuous values are not clipped. The default policy
rejects nonfinite continuous input. The explicit `invalid_policy: "skip"`
option excludes windows touching invalid rows and records their original row
indices; it does not impute, relabel, or compact the time axis.

An optional engineering-edge JSON file is a list with these fields:

```json
[
  {
    "src": "source_name",
    "dst": "receiver_name",
    "relation_type": "physical_flow",
    "source_document": "actual_document_identifier",
    "source_location": "actual_page_or_section",
    "evidence_status": "documented",
    "dataset_version": "actual_release_identifier"
  }
]
```

Replace all example values using real sources and node names. Exact duplicates
merge provenance; conflicting labels or attributes for one ordered pair are
rejected. The current matrix representation permits one type per ordered pair.
Engineering and self weights are 1; auxiliary weights are absolute Fit Pearson
coefficients. Pearson signs remain metadata, not causal direction. The default
stability checks require compatible signs and magnitudes on normal Fit
segments and normal Validation. Auxiliary links never overwrite engineering
labels. No implicit graph symmetrization is performed.

## Configuration

`configs/method.json` exposes all runnable defaults:

- window 100, stride 1, hidden width 128, four heads, two temporal and two
  spatial layers; feed-forward width 512, dropout 0;
- Adam, constant learning rate 0.0001, batch 64, maximum 30 epochs, seed 42;
  select the checkpoint with minimum normal Validation whole-window MSE;
- association weights 0.001, shared-gradient cap 0.1, node mixing weight 0.9;
- Pearson top-K 3, at least 100 valid pairs, absolute Fit correlation 0.7,
  segment/Validation magnitude 0.5, three Fit segments per recording;
- static POT starts at normal-score Q95, target upper-tail rate 0.01,
  minimum 50 strict exceedances; zero-location GPD maximum-likelihood fit.

Seed, graph eligibility/stability settings, feed-forward width, and numerical
POT settings are explicit implementation defaults. Set dataset-specific
values and split boundaries before evaluation and retain the configuration.
No hyperparameter search or test-label threshold selection is included.
POT failure stops calibration; it does not silently fall back to Q99. A zero or
numerically degenerate component variance makes the combined score unavailable.
For a declared reconstruction-only comparison, set `scoring.readout` to `R`.

## Run

From the extracted package directory:

```sh
python -m kgdiag train --config configs/method.json --schema data/nodes.json --fit data/fit.csv --validation data/validation.csv --calibration data/calibration.csv --edges data/engineering_edges.json --output runs/kgdiag --device cuda
python -m kgdiag infer --checkpoint runs/kgdiag/best.pt --data data/test.csv --output outputs/test --device cuda --evidence
```

Omit `--edges` if no engineering edges are available. The graph then contains
statistical auxiliary links and self connections, without invented physical
types. Omit `--calibration` if residual-rarity evidence is not needed; the
detection threshold is still fitted on normal Fit scores.

Training saves `best.pt`, the last completed epoch state, preprocessing, graph,
configuration, and source hashes in the requested output directory. An
interrupt saves a separate partial-epoch state. Existing output directories
must be empty, preventing accidental overwrites. The command does not silently
resume a partial epoch. Checkpoints are locally generated artifacts; use
trusted checkpoint files.

Inference produces endpoint-indexed `scores.csv` and, if requested, compact
node/type evidence. Each score uses only observations within its ending window.
The saved adjacency flag preserves recording breaks and unscored gaps. With
timestamp metadata, inference rejects queries preceding normal reference
availability unless explicitly marked `--retrospective`. Uninstantiated
relation types are flagged unavailable and must be excluded from rankings.

Optional evaluation is separate from training and calibration:

```sh
python -m kgdiag evaluate --scores outputs/test/scores.csv --labels data/test_with_labels.csv --column label --schema data/nodes.json --output outputs/test/metrics.json
```

The labels CSV must preserve exactly the observation row order, timestamps,
and recording identifiers. Labels are `0`, `1`, or `-1` for unknown. Unknowns
are excluded from binary metrics and break adjusted events. The evaluator
reports raw and point-adjusted metrics separately, uses continuous scores for
ROC-AUC and precision-recall measures, and reports **AP separately from
trapezoidal PR-AUC**. Point adjustment is confined to each contiguous positive
interval; it never changes the threshold or continuous score.

For retrospective diagnosis use `kgdiag.diagnostics.event_evidence`: average
scores over the specified event endpoints, then rank candidates. Set
`available_until` for a diagnosis restricted to an actual query time. Missing
targets and unsupported relation types remain unavailable, not zero-residual
normal observations. States without reconstruction targets can have graph
evidence but are not candidates for the mixed residual-based ranking.

## Code map

| File | Function |
|---|---|
| `data.py` | Fit-only preprocessing and boundary-safe windows |
| `graph.py` | Engineering/Pearson graph construction and provenance |
| `model.py` | Dual encoders, feature interaction, gated reconstruction |
| `association.py` | Supported probabilities, training roots, components |
| `training.py` | Separate gradients and shared auxiliary norm cap |
| `scoring.py` | Frozen standardization, static POT, residual rarity |
| `diagnostics.py` | Event-average evidence and coverage-aware rankings |
| `evaluation.py` | Isolated raw/PA metrics and continuous ranking measures |
| `cli.py` | Training and inference entry points |

## License and attribution

The authors retain a research-review license for KGDiag code; see `LICENSE`.
This is not an MIT release of KGDiag. Anomaly Transformer-derived components
retain their MIT notice in `licenses/AnomalyTransformer-MIT.txt`.
Dataset access and redistribution follow the providers' separate terms.
Please cite the associated KGDiag manuscript and the original work when using
or discussing these methods. See `THIRD_PARTY_NOTICES.md`.
