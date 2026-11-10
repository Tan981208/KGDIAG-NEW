"""Graph invariants and normal-only candidate-selection checks."""
import unittest

import numpy as np

from kgdiag.graph import build_graph


def schema(*kinds):
    return {"nodes": [{"name": f"x{i}", "kind": kind} for i, kind in enumerate(kinds)]}


def config(k=0):
    return dict(top_k=k, min_pair_observations=4, min_abs_correlation=0.25,
                min_segment_abs_correlation=0.2, fit_segment_count=3,
                engineering_weight=1.0, self_weight=1.0)


def edge(src="x0", dst="x1", relation="physical_flow", source="process.pdf"):
    return dict(src=src, dst=dst, relation_type=relation, source_document=source,
                source_location="page 2", evidence_status="document_supported")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-2.0, 3.0, 120)
        self.two = np.stack([self.x, 3 * self.x + 1], axis=1)

    def test_direction_self_and_prior(self):
        g = build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [edge()], config())
        np.testing.assert_array_equal(g["support"], [[True, False], [True, True]])
        np.testing.assert_allclose(g["prior"], [[1, 0], [0.5, 0.5]])
        self.assertEqual(g["relation_ids"][0, 1], -1)

    def test_duplicate_provenance_is_merged(self):
        g = build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [edge(), edge(source="second.pdf"), edge()], config())
        links = [e for e in g["edges"] if e["origin"] == "engineering"]
        self.assertEqual(len(links), 1)
        self.assertEqual(len(links[0]["provenance"]), 2)
        self.assertEqual(g["weights"][1, 0], 1)

    def test_conflicting_type_or_missing_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [edge(), edge(relation="control_logic")], config())
        bad = edge()
        del bad["source_document"]
        with self.assertRaisesRegex(ValueError, "source_document"):
            build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [bad], config())

    def test_constant_and_categorical_do_not_create_pearson_links(self):
        values = np.stack([self.x, np.ones_like(self.x), self.x], axis=1)
        g = build_graph(schema("continuous", "continuous", "categorical"),
                        values, values, [], config(k=2))
        np.testing.assert_array_equal(g["support"], np.eye(3, dtype=bool))

    def test_validation_sign_reversal_rejects_candidate(self):
        val = self.two.copy()
        val[:, 1] *= -1
        g = build_graph(schema("continuous", "continuous"), self.two, val, [], config(k=1))
        np.testing.assert_array_equal(g["support"], np.eye(2, dtype=bool))

    def test_fit_segment_sign_reversal_rejects_candidate(self):
        x = np.sin(np.arange(120) * 0.7)
        y = x.copy()
        y[90:] *= -1
        fit = np.stack([x, y], axis=1)
        val = np.stack([x, x], axis=1)
        g = build_graph(schema("continuous", "continuous"), fit, val, [], config(k=1),
                        fit_segments=[slice(0, 90), slice(90, 120)])
        np.testing.assert_array_equal(g["support"], np.eye(2, dtype=bool))

    def test_ties_use_registry_order(self):
        values = np.stack([self.x, self.x, self.x], axis=1)
        g = build_graph(schema(*(["continuous"] * 3)), values, values, [], config(k=1))
        self.assertTrue(g["support"][2, 0])
        self.assertFalse(g["support"][2, 1])
        np.testing.assert_allclose(g["prior"].sum(axis=1), 1)

    def test_engineering_not_overwritten_and_reverse_stays_auxiliary(self):
        g = build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [edge()], config(k=1))
        by_pair = {(e["dst"], e["src"]): e for e in g["edges"]}
        self.assertEqual(by_pair[("x1", "x0")]["relation_type"], "physical_flow")
        self.assertEqual(by_pair[("x0", "x1")]["relation_type"], "implicit_learned")

    def test_invalid_pairs_and_fixed_weight_guard(self):
        values = self.two.copy()
        values[3:, 1] = np.nan
        g = build_graph(schema("continuous", "continuous"), values, self.two, [], config(k=1))
        np.testing.assert_array_equal(g["support"], np.eye(2, dtype=bool))
        c = config()
        c["engineering_weight"] = 0.9
        with self.assertRaisesRegex(ValueError, "must equal 1"):
            build_graph(schema("continuous", "continuous"), self.two, self.two, [], c)

    def test_segments_and_declared_extra_type(self):
        with self.assertRaisesRegex(ValueError, "ordered"):
            build_graph(schema("continuous", "continuous"), self.two, self.two, [], config(),
                        fit_segments=[slice(0, 70), slice(60, 120)])
        c = config()
        c["additional_relation_types"] = ["documented_extra"]
        g = build_graph(schema("continuous", "continuous"), self.two, self.two,
                        [edge(relation="documented_extra")], c)
        self.assertIn("documented_extra", g["relation_names"])


if __name__ == "__main__":
    unittest.main()
