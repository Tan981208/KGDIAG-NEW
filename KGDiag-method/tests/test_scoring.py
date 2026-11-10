"""Normal-only calibration and evaluation properties on synthetic arrays."""
import unittest

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

from kgdiag.evaluation import evaluate
from kgdiag.scoring import combine, fit_pot, fit_standardization, residual_rarity


class StandardizationTests(unittest.TestCase):
    def setUp(self):
        self.fit = {"R": np.array([.1, .4, .7, .9]),
                    "DT": np.array([4., 5., 7., 10.]),
                    "DG": np.array([.5, .7, .8, 1.4])}

    def test_positive_affine_invariance_and_sign(self):
        statistics = fit_standardization(self.fit)
        query = {name: values[:2].copy() for name, values in self.fit.items()}
        base = combine(query, statistics)
        scales, offsets = {"R": 2., "DT": 10., "DG": .3}, {"R": 3., "DT": -20., "DG": 7.}
        changed_fit = {name: values * scales[name] + offsets[name] for name, values in self.fit.items()}
        changed_query = {name: values * scales[name] + offsets[name] for name, values in query.items()}
        np.testing.assert_allclose(combine(changed_query, fit_standardization(changed_fit)), base, atol=1e-12)
        for name, sign in (("R", 1), ("DT", -1), ("DG", 1)):
            altered = {key: value.copy() for key, value in query.items()}
            altered[name] += statistics[name]["std"]
            np.testing.assert_allclose(combine(altered, statistics) - base, sign, atol=1e-12)

    def test_self_loop_degeneracy_only_blocks_combined(self):
        fit = {**self.fit, "DG": np.zeros(4)}
        statistics = fit_standardization(fit)
        self.assertFalse(statistics["DG"]["usable"])
        with self.assertRaises(ValueError):
            combine(fit, statistics)
        np.testing.assert_array_equal(combine({"R": fit["R"]}, None, readout="R"), fit["R"])

    def test_reject_broadcasting_and_nonfinite_scales(self):
        statistics = fit_standardization(self.fit)
        invalid = [{**self.fit, "DT": self.fit["DT"][:, None]},
                   {**self.fit, "DG": self.fit["DG"][:1]},
                   {**self.fit, "R": [np.nan] * 4}]
        for values in invalid:
            with self.assertRaises(ValueError):
                combine(values, statistics)
            with self.assertRaises(ValueError):
                fit_standardization(values)
        for invalid_r in ([], [np.inf], [[1.]]):
            with self.assertRaises(ValueError):
                combine({"R": invalid_r}, None, readout="R")
        for value in (np.nan, np.inf, 0., -1.):
            corrupt = {name: dict(stat) for name, stat in statistics.items()}
            corrupt["DG"]["std"] = value
            with self.assertRaises(ValueError):
                combine(self.fit, corrupt)
        for extreme in ([1e308] * 4, [-1e308, 1e308, -1e308, 1e308]):
            with self.assertRaises(ValueError):
                fit_standardization({name: np.asarray(extreme) for name in self.fit})


class POTTests(unittest.TestCase):
    def test_exponential_tail_and_survival_consistency(self):
        # Deterministic quantiles of a normal-reference exponential distribution.
        # No anomaly labels or observed test scores participate in this fit.
        normal = -np.log1p(-(np.arange(10000) + .5) / 10000)
        fitted = fit_pot(normal, initial_quantile=.95, tail_probability=.01, min_exceedances=50)
        self.assertLess(abs(fitted["threshold"] - np.log(100.)), .1)
        self.assertLess(abs(fitted["shape"]), .05)
        conditional = genpareto.sf(fitted["threshold"] - fitted["initial_threshold"],
                                   fitted["shape"], loc=0, scale=fitted["scale"])
        self.assertAlmostEqual(conditional * fitted["exceedances"] / fitted["normal_count"], .01, places=7)

    def test_insufficient_tail_fails_without_fallback(self):
        with self.assertRaises(ValueError):
            fit_pot(np.arange(100.), min_exceedances=50)
        with self.assertRaises(ValueError):
            fit_pot(np.ones(1000))
        with self.assertRaises(ValueError):
            fit_pot(np.arange(1000.), min_exceedances=1.5)
        with self.assertRaises(ValueError):
            fit_pot(np.arange(1000.), tail_probability=np.nan)

    def test_upper_tail_ties_and_resolution(self):
        calibration = np.array([[1.], [2.], [2.], [3.]])
        values = np.array([[0.], [1.], [2.], [3.], [4.], [100.]])
        rarity = residual_rarity(calibration, values)
        np.testing.assert_allclose(rarity[:, 0], [0., 0., .2, .6, .8, .8])
        self.assertTrue(np.all(np.diff(rarity[:, 0]) >= 0))


class EvaluationTests(unittest.TestCase):
    def test_point_adjustment_respects_gaps_and_unknowns(self):
        labels = np.array([1, 1, 1, -1, 1, 1, 0])
        scores = np.array([2., 0., 0., 10., 0., 2., 0.])
        adjacent = np.array([False, True, False, True, True, True, True])
        old_scores, old_labels = scores.copy(), labels.copy()
        result = evaluate(labels, scores, 1., adjacent)
        self.assertEqual(result["raw"]["TP"], 2)
        self.assertEqual(result["raw"]["FN"], 3)
        self.assertEqual(result["point_adjusted"]["TP"], 4)
        self.assertEqual(result["point_adjusted"]["FN"], 1)
        self.assertEqual(result["known_points"], 6)
        self.assertEqual(result["unknown_points"], 1)
        np.testing.assert_array_equal(scores, old_scores)
        np.testing.assert_array_equal(labels, old_labels)
        self.assertAlmostEqual(result["AP"], average_precision_score(labels[labels >= 0], scores[labels >= 0]))

    def test_average_precision_is_distinct_from_trapezoidal_area(self):
        labels, scores = np.array([1, 0, 0, 1]), np.array([.1, .2, .3, .4])
        result = evaluate(labels, scores, .25)
        precision, recall, _ = precision_recall_curve(labels, scores)
        self.assertEqual(result["AP"], average_precision_score(labels, scores))
        self.assertEqual(result["PR_AUC_trapezoidal"], auc(recall, precision))
        self.assertNotEqual(result["AP"], result["PR_AUC_trapezoidal"])

    def test_strict_alarm_and_invalid_inputs(self):
        exact = evaluate([0, 1], [.5, .5], .5)
        self.assertEqual(exact["raw"]["TP"], 0)
        self.assertEqual(exact["raw"]["FP"], 0)
        for labels, scores, threshold, adjacent in (
                ([], [], 1., None), ([0, 1], [0., 1.], np.nan, None),
                ([0, 1], [0., 1.], np.inf, None), ([0, 2], [0., 1.], .5, None),
                ([0, np.nan], [0., 1.], .5, None), ([0, 1], [0., np.nan], .5, None),
                ([0, 1], [0., 1.], .5, [True, True]),
                ([0, 1], [0., 1.], .5, [False, np.nan]),
                ([0, 1], [0., 1.], .5, [False, 2])):
            with self.assertRaises(ValueError):
                evaluate(labels, scores, threshold, adjacent)


if __name__ == "__main__":
    unittest.main()
