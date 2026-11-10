"""Portable reference network for the current KGDiag method description.

This module implements the manuscript's current reconstruction architecture.

The temporal association design follows Anomaly Transformer (Xu et al., ICLR
2022), including its circular token convolution, encoder layout, and bounded
Gaussian-width transform. Upstream: https://github.com/thuml/Anomaly-Transformer
Copyright (c) 2022 THUML @ Tsinghua University, MIT License; see the accompanying
third-party notice. KGDiag adds typed graph associations, spatial pooling,
parallel cross-branch feature exchange, gated fusion, and a continuous decoder.

Graph matrices use [receiver, source] order. The network is current-inclusive
window reconstruction: it is not masked prediction or explicit lagged reading.
Only endpoint scores are prospective with respect to the window endpoint.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MODEL_CONFIG = {
    "window": 100,
    "d_model": 128,
    "heads": 4,
    "temporal_layers": 2,
    "spatial_layers": 2,
    "d_ff": 512,
    "dropout": 0.0,
}


def _sigma(value: torch.Tensor) -> torch.Tensor:
    """The bounded, input-conditioned width used in both association branches."""
    return torch.pow(3.0, torch.sigmoid(5.0 * value) + 1e-5) - 1.0


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    # [..., queries, d] -> [..., heads, queries, d/heads]
    return x.reshape(*x.shape[:-1], heads, x.shape[-1] // heads).transpose(-3, -2)


class TemporalLayer(nn.Module):
    """Temporal self-attention, Gaussian prior, and post-norm feed-forward block."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.sigma_projection = nn.Linear(d, heads)
        self.output = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, ff, kernel_size=1)
        self.conv2 = nn.Conv1d(ff, d, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, distance_squared: torch.Tensor):
        q = _split_heads(self.query(hidden), self.heads)
        k = _split_heads(self.key(hidden), self.heads)
        v = _split_heads(self.value(hidden), self.heads)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        series = torch.softmax(logits, dim=-1)
        width = _sigma(self.sigma_projection(hidden)).transpose(1, 2)
        # Row-wise Gaussian normalization cancels the common 1/(sqrt(2*pi)*sigma)
        # factor. Softmax avoids underflow of the entire unnormalized row.
        prior_logits = -distance_squared[None, None] / (2.0 * width.unsqueeze(-1).square())
        prior = torch.softmax(prior_logits, dim=-1)
        attended = torch.matmul(self.dropout(series), v)
        attended = attended.transpose(1, 2).contiguous().reshape_as(hidden)
        hidden = self.norm1(hidden + self.dropout(self.output(attended)))
        ff = self.dropout(F.gelu(self.conv1(hidden.transpose(1, 2))))
        ff = self.dropout(self.conv2(ff).transpose(1, 2))
        return self.norm2(hidden + ff), series, prior, width


class SpatialLayer(nn.Module):
    """Shared K/V attention, relation bias, and input-conditioned KG prior."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float, relation_count: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.sigma_projection = nn.Linear(d, heads)
        self.relation_bias = nn.Embedding(relation_count, heads)
        nn.init.zeros_(self.relation_bias.weight)
        self.output = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, support, base_prior, relation_ids):
        # hidden [B,L,N,d]; all attention distributions [B,L,H,N,N].
        q = _split_heads(self.query(hidden), self.heads)
        k = _split_heads(self.key(hidden), self.heads)
        v = _split_heads(self.value(hidden), self.heads)
        # -1 marks absence of a graph edge. The lookup uses a safe index; the
        # attention support mask removes those positions exactly afterwards.
        relation_bias = self.relation_bias(relation_ids.clamp_min(0)).permute(2, 0, 1)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + relation_bias[None, None]
        mask = support[None, None, None]
        series = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        exponent = _sigma(self.sigma_projection(hidden)).transpose(-1, -2)
        prior_logits = base_prior.clamp_min(1e-8).log()[None, None, None] * exponent.unsqueeze(-1)
        prior = torch.softmax(prior_logits.masked_fill(~mask, -torch.inf), dim=-1)
        attended = torch.matmul(self.dropout(series), v)
        attended = attended.transpose(-3, -2).contiguous().reshape_as(hidden)
        attended = self.dropout(F.gelu(self.output(attended)))
        hidden = self.norm1(hidden + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden, series, prior, exponent


class KGDiag(nn.Module):
    """KGDiag network with explicit schema and frozen graph tensors.

    ``schema`` fields:
      names: physical node names, all unique;
      continuous_indices: ordered node indices of the reconstruction targets;
      categorical_cardinalities: one entry per node, 0 for continuous nodes and
          K>=1 for a categorical node with known IDs 1..K and unknown ID 0;
      node_type_ids, node_stage_ids, node_family_ids: nonnegative metadata IDs.

    ``graph`` fields:
      support, prior, relation_ids: [N,N], receiver rows/source columns;
      relation_names: the actual relation vocabulary, including SELF.

    The temporal input consists of ordered continuous targets followed by one-hot
    categorical channels in physical node order (unknown=all-zero). The supplied
    graph node values/category IDs retain N physical nodes, not one-hot channels.
    """

    def __init__(self, model_config: dict[str, Any], schema: dict[str, Any], graph: dict[str, Any]):
        super().__init__()
        unknown = set(model_config) - set(DEFAULT_MODEL_CONFIG)
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        cfg = {**DEFAULT_MODEL_CONFIG, **model_config}
        for name in ("window", "d_model", "heads", "temporal_layers", "spatial_layers", "d_ff"):
            if int(cfg[name]) != cfg[name] or cfg[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
            cfg[name] = int(cfg[name])
        # The released manuscript profile uses no attention dropout. This avoids
        # treating post-dropout, non-normalized routing weights as distributions.
        if float(cfg["dropout"]) != 0.0:
            raise ValueError("This reference profile requires dropout=0 to keep routing distributions exact")
        d, heads, ff = cfg["d_model"], cfg["heads"], cfg["d_ff"]
        if d % heads or d % 2:
            raise ValueError("d_model must be even and divisible by heads")
        self.config = cfg
        self.schema = schema
        self.window = cfg["window"]
        names = schema["names"]
        self.n_nodes = len(names)
        if self.n_nodes == 0 or len(set(names)) != self.n_nodes:
            raise ValueError("Node names must be nonempty and unique")
        targets = [int(i) for i in schema["continuous_indices"]]
        if not targets or len(set(targets)) != len(targets) or min(targets) < 0 or max(targets) >= self.n_nodes:
            raise ValueError("continuous_indices must contain distinct valid target indices")
        cardinalities = [int(k) for k in schema["categorical_cardinalities"]]
        if len(cardinalities) != self.n_nodes or any(k < 0 for k in cardinalities):
            raise ValueError("Invalid per-node categorical cardinalities")
        if {i for i, k in enumerate(cardinalities) if k == 0} != set(targets):
            raise ValueError("Continuous targets must be exactly the nodes with zero categorical cardinality")
        self.cardinalities = cardinalities
        self.n_targets = len(targets)
        self.input_dim = self.n_targets + sum(cardinalities)
        self.register_buffer("continuous_indices", torch.tensor(targets, dtype=torch.long))

        support = torch.as_tensor(graph["support"], dtype=torch.bool).clone()
        prior = torch.as_tensor(graph["prior"], dtype=torch.float32).clone()
        relation_ids = torch.as_tensor(graph["relation_ids"], dtype=torch.long).clone()
        relation_names = list(graph["relation_names"])
        if not relation_names or len(set(relation_names)) != len(relation_names):
            raise ValueError("relation_names must be nonempty and unique")
        if any(tuple(x.shape) != (self.n_nodes, self.n_nodes) for x in (support, prior, relation_ids)):
            raise ValueError("All graph tensors must use the physical node order and shape [N,N]")
        if not bool(support.diagonal().all()):
            raise ValueError("Every node must retain a computational self connection")
        if not bool(torch.isfinite(prior).all()) or bool((prior[support] <= 0).any()) or bool((prior[~support] != 0).any()):
            raise ValueError("Graph prior must be finite, positive on support, and zero off support")
        if not torch.allclose(prior.sum(-1), torch.ones(self.n_nodes), atol=2e-6, rtol=2e-6):
            raise ValueError("Graph prior rows must sum to one")
        supported_ids = relation_ids[support]
        if int(supported_ids.min()) < 0 or int(supported_ids.max()) >= len(relation_names):
            raise ValueError("Supported relation_ids exceed the explicit relation vocabulary")
        relation_ids = relation_ids.masked_fill(~support, -1)
        self.relation_names = relation_names
        self.register_buffer("graph_support", support)
        self.register_buffer("graph_base_prior", prior)
        self.register_buffer("graph_relation_ids", relation_ids)

        self.temporal_embedding = nn.Conv1d(self.input_dim, d, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.temporal_embedding.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(self.window, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        positional = torch.zeros(self.window, d)
        positional[:, 0::2] = torch.sin(position * frequency)
        positional[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("temporal_position", positional)
        time_index = torch.arange(self.window, dtype=torch.float32)
        self.register_buffer("distance_squared", (time_index[:, None] - time_index[None, :]).square())
        self.temporal_layers = nn.ModuleList([TemporalLayer(d, heads, ff, 0.0) for _ in range(cfg["temporal_layers"])])
        self.temporal_norm = nn.LayerNorm(d)

        self.scalar_embedding = nn.Linear(1, d)
        self.category_embeddings = nn.ModuleDict({str(i): nn.Embedding(k + 1, d, padding_idx=0)
                                                 for i, k in enumerate(cardinalities) if k > 0})
        self.spatial_time_embedding = nn.Embedding(self.window, d)
        self.metadata_embeddings = nn.ModuleDict()
        for role in ("type", "stage", "family"):
            ids = torch.as_tensor(schema[f"node_{role}_ids"], dtype=torch.long).clone()
            if tuple(ids.shape) != (self.n_nodes,) or bool((ids < 0).any()):
                raise ValueError(f"Invalid node_{role}_ids")
            self.register_buffer(f"node_{role}_ids", ids)
            self.metadata_embeddings[f"node_{role}"] = nn.Embedding(int(ids.max()) + 1, d)
        self.spatial_layers = nn.ModuleList([SpatialLayer(d, heads, ff, 0.0, len(relation_names))
                                             for _ in range(cfg["spatial_layers"])])
        self.pool_projection = nn.Linear(d, 1)
        self.cross_temporal = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.cross_spatial = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.cross_temporal_norm = nn.LayerNorm(d)
        self.cross_spatial_norm = nn.LayerNorm(d)
        self.temporal_gate = nn.Linear(d, d)
        self.spatial_gate = nn.Linear(d, d)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, 4 * d), nn.ReLU(),
            nn.Linear(4 * d, self.n_targets),
        )

    def forward(self, temporal_input, node_values, category_ids):
        """Return reconstruction and full-support association lists for scoring."""
        if temporal_input.ndim != 3 or tuple(temporal_input.shape[1:]) != (self.window, self.input_dim):
            raise ValueError(f"temporal_input must have shape [B,{self.window},{self.input_dim}]")
        expected = (temporal_input.shape[0], self.window, self.n_nodes)
        if tuple(node_values.shape) != expected or tuple(category_ids.shape) != expected:
            raise ValueError("node_values/category_ids shape does not match the physical-node schema")
        if category_ids.dtype != torch.long:
            raise ValueError("category_ids must use torch.long, with unknown category ID 0")
        temporal = self.temporal_embedding(temporal_input.transpose(1, 2)).transpose(1, 2)
        temporal = temporal + self.temporal_position[None]
        temporal_series, temporal_prior, temporal_sigma = [], [], []
        for layer in self.temporal_layers:
            temporal, series, prior, width = layer(temporal, self.distance_squared)
            temporal_series.append(series)
            temporal_prior.append(prior)
            temporal_sigma.append(width)
        temporal = self.temporal_norm(temporal)

        embedded_nodes = []
        for i, cardinality in enumerate(self.cardinalities):
            if cardinality:
                ids = category_ids[:, :, i]
                if bool(((ids < 0) | (ids > cardinality)).any()):
                    raise ValueError(f"Categorical node {i} contains an out-of-vocabulary ID")
                # Explicit masking guarantees an unknown value contributes zero
                # even if a loaded embedding has a nonzero padding row.
                node = self.category_embeddings[str(i)](ids) * (ids != 0).unsqueeze(-1)
            else:
                node = self.scalar_embedding(node_values[:, :, i:i + 1])
            embedded_nodes.append(node)
        spatial = torch.stack(embedded_nodes, dim=2)
        time_ids = torch.arange(self.window, device=temporal_input.device)
        spatial = spatial + self.spatial_time_embedding(time_ids)[None, :, None]
        for role, embedding in self.metadata_embeddings.items():
            spatial = spatial + embedding(getattr(self, f"{role}_ids"))[None, None]
        graph_series, graph_prior, graph_sigma = [], [], []
        for layer in self.spatial_layers:
            spatial, series, prior, exponent = layer(spatial, self.graph_support, self.graph_base_prior, self.graph_relation_ids)
            graph_series.append(series)
            graph_prior.append(prior)
            graph_sigma.append(exponent)
        pool_weights = torch.softmax(self.pool_projection(spatial).squeeze(-1), dim=-1)
        pooled = (pool_weights.unsqueeze(-1) * spatial).sum(dim=2)

        # Both cross-attentions read the original encoder outputs. In particular,
        # the reverse direction does NOT read the already-updated temporal branch.
        context_t = self.cross_temporal(temporal, pooled, pooled, need_weights=False)[0]
        context_g = self.cross_spatial(pooled, temporal, temporal, need_weights=False)[0]
        updated_t = self.cross_temporal_norm(temporal + context_t)
        updated_g = self.cross_spatial_norm(pooled + context_g)
        gate = torch.sigmoid(self.temporal_gate(updated_t) + self.spatial_gate(updated_g))
        fused = gate * updated_t + (1.0 - gate) * updated_g
        return {
            "reconstruction": self.reconstruction_head(fused),
            "support": self.graph_support,
            "temporal_series": temporal_series, "temporal_prior": temporal_prior,
            "graph_series": graph_series, "graph_prior": graph_prior,
            "temporal_sigma": temporal_sigma, "graph_sigma": graph_sigma,
            "graph_pool_weights": pool_weights,
        }
