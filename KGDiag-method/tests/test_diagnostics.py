import unittest
import numpy as np
from kgdiag.diagnostics import event_evidence, rank_candidates


class DiagnosticsTests(unittest.TestCase):
    def test_coverage_ties_and_mean_before_rank(self):
        summary = rank_candidates([2,2,0], ["a","b","missing"], [True,True,False])
        self.assertEqual([r["rank"] for r in summary["ranking"]], [1.5,1.5])
        self.assertEqual(summary["unavailable"], ["missing"])
        out = event_evidence([2,3], [[100,0],[0,1]], [[1,0],[2,0]],2,4,["a","b"],["self","unobserved"],[True,False])
        self.assertEqual(out["variables"]["ranking"][0]["name"], "a")
        self.assertEqual(out["variables"]["ranking"][0]["score"],50)

    def test_alarm_time_cannot_read_future_rows(self):
        node=np.array([[1,0],[0,100.]])
        out=event_evidence([2,3],node,np.ones((2,1)),2,4,["a","b"],["self"],[True],available_until=2)
        self.assertEqual(out["last_endpoint"],2)
        self.assertEqual(out["variables"]["ranking"][0]["name"],"a")
        self.assertEqual(event_evidence([2,3],node,np.ones((2,1)),0,1,["a","b"],["self"],[True])["scored_endpoints"],0)


if __name__ == "__main__":
    unittest.main()
