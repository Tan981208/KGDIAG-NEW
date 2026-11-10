"""Synthetic checks of the model's public tensor and information-flow contracts."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from kgdiag.model import KGDiag
from kgdiag.graph import build_graph


def example():
    config = dict(window=9, d_model=16, heads=4, temporal_layers=2,
                  spatial_layers=2, d_ff=32, dropout=0.0)
    schema = dict(names=["sensor_a", "sensor_b", "state"],
                  continuous_indices=[0, 1], categorical_cardinalities=[0, 0, 2],
                  node_type_ids=[0, 0, 1], node_stage_ids=[0, 0, 0],
                  node_family_ids=[0, 0, 1])
    # source 0 can inform receiver 1, but receiver 0 cannot read source 1.
    support = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=torch.bool)
    prior = support.float() / support.sum(-1, keepdim=True)
    relation_ids = torch.tensor([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    graph = dict(support=support, prior=prior, relation_ids=relation_ids,
                 relation_names=["SELF", "physical_flow"])
    return config, schema, graph


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.config, self.schema, self.graph = example()
        self.model = KGDiag(self.config, self.schema, self.graph)
        self.temporal = torch.randn(2, 9, 4)
        self.nodes = torch.randn(2, 9, 3)
        self.categories = torch.zeros(2, 9, 3, dtype=torch.long)
        self.categories[:, :, 2] = 1

    def run_model(self):
        return self.model(self.temporal, self.nodes, self.categories)

    def test_shapes_support_probability_and_backward(self):
        result = self.run_model()
        self.assertEqual(result["reconstruction"].shape, (2, 9, 2))
        self.assertEqual(result["temporal_series"][0].shape, (2, 4, 9, 9))
        self.assertEqual(result["graph_series"][0].shape, (2, 9, 4, 3, 3))
        for name in ("temporal_series", "temporal_prior", "graph_series", "graph_prior"):
            for value in result[name]:
                self.assertTrue(torch.isfinite(value).all())
                torch.testing.assert_close(value.sum(-1), torch.ones_like(value.sum(-1)), atol=2e-6, rtol=2e-6)
                if name.startswith("graph"):
                    self.assertTrue((value[..., ~self.graph["support"]] == 0).all())
                    self.assertTrue((value[..., 0, 0] == 1).all())
                    self.assertTrue((value[..., 1, 0] > 0).all())
                    self.assertTrue((value[..., 0, 1] == 0).all())
        result["reconstruction"].square().mean().backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_unknown_category_and_batch_invariance(self):
        self.model.eval()
        self.categories[:, :, 2] = 0
        with torch.no_grad():
            expected = self.run_model()["reconstruction"]
            self.model.category_embeddings["2"].weight[0].fill_(1000)
            actual = self.run_model()["reconstruction"]
            single = self.model(self.temporal[:1], self.nodes[:1], self.categories[:1])["reconstruction"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(single, expected[:1], atol=2e-5, rtol=2e-5)

    def test_parallel_cross_attention_and_decoder(self):
        captured = {}
        h1 = self.model.temporal_norm.register_forward_hook(
            lambda module, inputs, output: captured.update(original_temporal=output.detach().clone()))
        h2 = self.model.cross_spatial.register_forward_pre_hook(
            lambda module, inputs: captured.update(reverse_key=inputs[1].detach().clone()))
        self.run_model()
        h1.remove()
        h2.remove()
        torch.testing.assert_close(captured["reverse_key"], captured["original_temporal"], atol=0, rtol=0)
        linear = [m for m in self.model.reconstruction_head if isinstance(m, nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linear], [(16, 64), (64, 64), (64, 2)])
        self.assertEqual(sum(isinstance(m, nn.ReLU) for m in self.model.reconstruction_head), 2)
        for layer in self.model.spatial_layers:
            self.assertIsInstance(layer.key, nn.Linear)
            self.assertIsInstance(layer.value, nn.Linear)
            self.assertEqual(layer.relation_bias.num_embeddings, 2)
            self.assertFalse(any("adapter" in name for name, _ in layer.named_parameters()))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.temporal[:, :-1], self.nodes, self.categories)
        with self.assertRaises(ValueError):
            self.model(self.temporal, self.nodes, self.categories.float())
        self.categories[:, :, 2] = 3
        with self.assertRaises(ValueError):
            self.run_model()
        bad = copy.deepcopy(self.schema)
        bad["continuous_indices"] = [0, 2]
        with self.assertRaises(ValueError):
            KGDiag(self.config, bad, self.graph)
        bad_graph = copy.deepcopy(self.graph)
        bad_graph["support"][0, 0] = False
        with self.assertRaises(ValueError):
            KGDiag(self.config, self.schema, bad_graph)
        with self.assertRaises(ValueError):
            KGDiag({**self.config, "unknown_option": 1}, self.schema, self.graph)

    def test_graph_builder_absent_relation_ids(self):
        registry = dict(nodes=[dict(name=name, kind="continuous" if i < 2 else "categorical")
                               for i, name in enumerate(self.schema["names"])])
        raw = np.random.default_rng(2).normal(size=(40, 3))
        graph_config = dict(top_k=0, min_pair_observations=5, min_abs_correlation=0.9,
                            min_segment_abs_correlation=0.8, fit_segment_count=2,
                            engineering_weight=1.0, self_weight=1.0)
        graph = build_graph(registry, raw[:30], raw[30:], [], graph_config)
        self.assertTrue((graph["relation_ids"][~graph["support"]] == -1).all())
        model = KGDiag(self.config, self.schema, graph)
        result = model(self.temporal, self.nodes, self.categories)
        self.assertTrue(torch.isfinite(result["reconstruction"]).all())
        self.assertTrue((model.graph_relation_ids[~model.graph_support] == -1).all())
        for key in ("graph_series", "graph_prior"):
            for value in result[key]:
                torch.testing.assert_close(value, torch.eye(3).expand_as(value), atol=0, rtol=0)
        graph["relation_ids"][0, 0] = -1
        with self.assertRaises(ValueError):
            KGDiag(self.config, self.schema, graph)


if __name__ == "__main__":
    unittest.main()
