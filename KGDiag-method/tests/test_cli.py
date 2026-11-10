"""Tiny synthetic CPU integration test; temporary files are removed automatically."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kgdiag.cli import infer, load_checkpoint, metrics, train, write_json


class PipelineTest(unittest.TestCase):
    def test_train_infer_and_aligned_evaluation(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(52)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema = dict(nodes=[dict(name="a", kind="continuous"), dict(name="b", kind="continuous"),
                                 dict(name="state", kind="categorical")], timestamp_column="time", max_gap_seconds=1.5)
            write_json(root / "nodes.json", schema)
            # Disjoint normal splits and future test; artificial values only.
            start = 0
            for name, count in [("fit", 320), ("val", 96), ("cal", 96), ("test", 96)]:
                a = rng.normal(size=count)
                frame = pd.DataFrame(dict(time=pd.date_range("2000-01-01", periods=start+count, freq="s")[start:],
                                          a=a, b=0.9*a + 0.1*rng.normal(size=count), state=np.arange(count)%2))
                if name == "test":
                    frame["label"] = 0
                    frame.loc[25:35, "label"] = 1
                    frame.loc[25:35, "a"] += 3
                frame.to_csv(root / f"{name}.csv", index=False)
                start += count
            cfg = dict(model=dict(window=8,d_model=8,heads=2,temporal_layers=1,spatial_layers=1,d_ff=16,dropout=0.),
                       training=dict(epochs=1,batch_size=32,learning_rate=.001,seed=42,lambda_t=.001,lambda_g=.001,gradient_cap=.1),
                       data=dict(stride=1,invalid_policy="error"),
                       graph=dict(top_k=1,min_pair_observations=10,min_abs_correlation=.5,
                                  min_segment_abs_correlation=.2,fit_segment_count=2,engineering_weight=1.,self_weight=1.),
                       scoring=dict(readout="combined",pot=dict(initial_quantile=.8,tail_probability=.1,min_exceedances=10)),
                       diagnosis=dict(eta=.9))
            write_json(root / "config.json", cfg)
            arguments = argparse.Namespace(config=root/"config.json",schema=root/"nodes.json",fit=root/"fit.csv",
                validation=root/"val.csv",calibration=root/"cal.csv",edges=None,device="cpu",output=root/"run")
            with contextlib.redirect_stdout(io.StringIO()):
                train(arguments)
            saved = load_checkpoint(root/"run/best.pt")
            self.assertEqual(saved["epoch"], 1)
            self.assertIn("pot", saved)
            self.assertTrue(all(saved["standardization"][name]["usable"] for name in ["R", "DT", "DG"]))
            infer(argparse.Namespace(checkpoint=root/"run/best.pt",data=root/"test.csv",output=root/"infer",
                                     device="cpu",evidence=True,retrospective=False))
            scores = pd.read_csv(root/"infer/scores.csv")
            self.assertEqual(scores["row_index"].iloc[0], 7)
            self.assertEqual(len(scores), 89)
            self.assertTrue(scores["reference_available"].all())
            with np.load(root/"infer/evidence.npz") as evidence:
                self.assertEqual(evidence["mixed_node"].shape, (89, 2))
                np.testing.assert_allclose(evidence["relation"].sum(-1), 3/10 * scores["DG"].to_numpy(), rtol=2e-5)
            metrics(argparse.Namespace(scores=root/"infer/scores.csv",labels=root/"test.csv",column="label",
                                        schema=root/"nodes.json",output=root/"metrics.json"))
            # A query preceding the reference boundary is not silently called online.
            with self.assertRaisesRegex(ValueError, "precede"):
                infer(argparse.Namespace(checkpoint=root/"run/best.pt",data=root/"fit.csv",output=root/"earlier",
                                         device="cpu",evidence=False,retrospective=False))
            # Reordered labels fail their timestamp identity check.
            bad = pd.read_csv(root/"test.csv").iloc[::-1]
            bad.to_csv(root/"bad.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Timestamp mismatch"):
                metrics(argparse.Namespace(scores=root/"infer/scores.csv",labels=root/"bad.csv",column="label",
                                            schema=root/"nodes.json",output=root/"bad.json"))


if __name__ == "__main__":
    unittest.main()
