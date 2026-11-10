# Third-party attribution

The temporal association design follows:

Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long.
**Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy.**
ICLR 2022.

- Official code: https://github.com/thuml/Anomaly-Transformer
- Paper: https://openreview.net/forum?id=LzQQ89U1qm_
- Copyright (c) 2022 THUML @ Tsinghua University.
- MIT permission notice: `licenses/AnomalyTransformer-MIT.txt`.

The temporal circular embedding, association attention, Gaussian-width transform,
and encoder structure in `kgdiag/model.py` follow that implementation. KGDiag's
graph path, feature fusion, gradient assembly, and diagnostic definitions are
identified in the source modules. The upstream license remains applicable to
the upstream-derived portions; the package-level research-review terms do not
replace it.

PyTorch, NumPy, pandas, SciPy, and scikit-learn are installed as dependencies,
not vendored in this archive. They retain their own licenses. POT fitting uses
SciPy's generalized Pareto maximum-likelihood implementation; no third-party
SPOT code is bundled. Dataset licenses and access links are in `DATASETS.md`.
