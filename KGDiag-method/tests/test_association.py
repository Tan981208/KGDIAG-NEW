"""Synthetic property checks; no real observations, artifacts, or training run."""
import unittest

import torch
from torch import nn

from kgdiag.association import components, edge_jeffreys, loss_terms, normalize_supported
from kgdiag.training import capped_backward


def small_output():
    b, length, heads, nodes, targets = 2, 3, 2, 3, 2
    support = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.bool)
    s = torch.tensor([[.8, .2, 0], [0, 1., 0], [.2, .3, .5]], dtype=torch.float64)
    p = torch.tensor([[.5, .5, 0], [0, 1., 0], [.6, .3, .1]], dtype=torch.float64)
    graph_s = s.expand(b, length, heads, nodes, nodes).clone()
    graph_p = p.expand(b, length, heads, nodes, nodes).clone()
    temporal_s = torch.full((b, heads, length, length), 1 / length, dtype=torch.float64)
    temporal_p = torch.tensor([.2, .3, .5], dtype=torch.float64).expand_as(temporal_s).clone()
    prediction = torch.ones(b, length, targets, dtype=torch.float64)
    return {"reconstruction": prediction, "temporal_series": [temporal_s, temporal_s.clone()],
            "temporal_prior": [temporal_p, temporal_p.clone()],
            "graph_series": [graph_s, graph_s.clone()], "graph_prior": [graph_p, graph_p.clone()],
            "support": support}, torch.zeros_like(prediction)


class AssociationTests(unittest.TestCase):
    def test_probability_and_support(self):
        mask = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool)
        values = torch.tensor([[2., 3., 100.], [40., 7., 20.]], dtype=torch.float64)
        normalized = normalize_supported(values, mask)
        self.assertTrue(torch.allclose(normalized.sum(-1), torch.ones(2, dtype=torch.float64)))
        self.assertEqual(int(torch.count_nonzero(normalized[~mask])), 0)
        self.assertEqual(float(normalized[1, 1]), 1.)
        with self.assertRaises(ValueError):
            normalize_supported(values, torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            normalize_supported(torch.zeros_like(values), mask)

    def test_edge_and_type_conservation(self):
        output, target = small_output()
        result = components(output, target, output["support"])
        self.assertTrue(bool((result["edge"] >= 0).all()))
        self.assertEqual(int(torch.count_nonzero(result["edge"][:, ~output["support"]])), 0)
        self.assertEqual(int(torch.count_nonzero(result["node_graph"][:, 1])), 0)
        self.assertGreater(float(result["edge"][0, 0, 1]), 0.)
        relation_ids = torch.tensor([[0, 1, -1], [-1, 0, -1], [1, 2, 0]])
        grouped = torch.stack([(result["edge"] * (relation_ids == rid)).sum((-2, -1)) for rid in (0, 1, 2)])
        self.assertTrue(torch.allclose(grouped.sum(0), result["node_graph"].sum(-1)))
        self.assertTrue(torch.allclose(grouped.sum(0), 3.0 * result["DG"] / 20.0))
        self.assertTrue(torch.equal(result["R"], result["residual"].mean(-1)))

    def test_layer_reduction_and_endpoint(self):
        output, target = small_output()
        two = components(output, target, output["support"])
        one = dict(output)
        for key in ("temporal_series", "temporal_prior", "graph_series", "graph_prior"):
            one[key] = output[key][:1]
        first = components(one, target, output["support"])
        self.assertTrue(torch.allclose(two["DT"], 2 * first["DT"]))
        self.assertTrue(torch.allclose(two["DG"], 2 * first["DG"]))
        self.assertTrue(torch.allclose(two["edge"], first["edge"]))
        for key in ("graph_series", "graph_prior"):
            one[key] = [v.clone() for v in one[key]]
        one["graph_series"][0][:, :-1] = one["graph_prior"][0][:, :-1]
        self.assertTrue(torch.equal(components(one, target, output["support"])["DG"], first["DG"]))
        two_loss, one_loss = loss_terms(output, target), loss_terms({**output,
            "temporal_series": output["temporal_series"][:1], "temporal_prior": output["temporal_prior"][:1],
            "graph_series": output["graph_series"][:1], "graph_prior": output["graph_prior"][:1]}, target)
        for key in two_loss:
            self.assertTrue(torch.equal(two_loss[key], one_loss[key]))

    def test_equal_forward_distinct_detached_gradients(self):
        s = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([.6, .4], dtype=torch.float64, requires_grad=True)
        ser = edge_jeffreys(s, p.detach()).sum()
        pri = edge_jeffreys(s.detach(), p).sum()
        self.assertEqual(float(ser), float(pri))
        gs = torch.autograd.grad(ser, (s, p), allow_unused=True)
        gp = torch.autograd.grad(pri, (s, p), allow_unused=True)
        self.assertGreater(float(gs[0].norm()), 0.)
        self.assertIsNone(gs[1])
        self.assertIsNone(gp[0])
        self.assertGreater(float(gp[1].norm()), 0.)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Parameter(torch.tensor([1., 2.]))
        self.rec_only = nn.Parameter(torch.tensor([1.]))
        self.assoc_only = nn.Parameter(torch.tensor([1.]))
        self.unused = nn.Parameter(torch.tensor([1.]))


class GradientCapTests(unittest.TestCase):
    def run_case(self, direction, zero_rec=False, shared=True):
        model = ToyModel()
        rec = model.rec_only.sum() + (model.shared.square().sum() * (0. if zero_rec else 1.) if shared else 0.)
        assoc = model.assoc_only.sum() * 7 + (model.shared.sum() * direction if shared else 0.)
        parts = {"reconstruction": rec, "temporal_series": assoc * 0,
                 "temporal_prior": assoc, "graph_series": assoc * 0, "graph_prior": assoc * 0}
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        stats = capped_backward(model, parts, lambda_t=1., lambda_g=0., cap=.1)
        self.assertEqual(stats["bound_passed"], 1.)
        self.assertLessEqual(stats["norm_A_after"], .1 * stats["norm_R_shared"])
        self.assertLessEqual(stats["norm_effective_A_after"], .1 * stats["norm_R_shared"])
        self.assertEqual(float(model.rec_only.grad), 2.)
        self.assertEqual(float(model.assoc_only.grad), 7.)
        self.assertIsNone(model.unused.grad)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), initial[name]))
        return model, stats

    def test_aligned_and_opposed(self):
        for direction in (-100., 100.):
            _, stats = self.run_case(direction)
            self.assertEqual(stats["clipped"], 1.)

    def test_zero_reconstruction(self):
        model, stats = self.run_case(100., zero_rec=True)
        self.assertEqual(stats["coefficient"], 0.)
        self.assertEqual(int(torch.count_nonzero(model.shared.grad)), 0)

    def test_empty_shared(self):
        model, stats = self.run_case(100., shared=False)
        self.assertEqual(stats["shared_parameter_count"], 0.)
        self.assertIsNone(model.shared.grad)


if __name__ == "__main__":
    unittest.main()
