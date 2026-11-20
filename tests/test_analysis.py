import tempfile
import unittest
from pathlib import Path
from glue.analyze_glue import points, score, summarize, read


class AnalysisTests(unittest.TestCase):
    def test_constant_prediction(self):
        r=[dict(idx=i,gold=g,pred=1,a=-1,b=1) for i,g in enumerate((0,0,1))]
        metrics,_=score(r,("a","b"))
        self.assertEqual(metrics["mcc"],0)
        self.assertEqual(metrics["recall_0"],0)
        self.assertEqual(metrics["predicted_0"],0)
        self.assertEqual(metrics["f1_positive"],.5)
        self.assertEqual(metrics["confusion"],[[0,2,0],[0,1,0]])

    def test_ties_and_legacy_loss(self):
        r=[dict(idx=0,gold=2,pred=-1,a=-1,b=2,c=2)]
        metrics,_=score(r,("a","b","c"))
        self.assertEqual((metrics["correct"],metrics["ties"]),(0,1))
        old=[dict(step="1",loss="2.4")]
        self.assertEqual(points(old,"loss","wall_s"),([],[]))
        self.assertEqual(points(old,"loss","step"),([1.],[2.4]))

    def test_empty_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            self.assertEqual(summarize(p/"out",p/"empty"),{})
            self.assertEqual(sum(r["task"]=="mrpc" for r in read(p/"out"/"per_seed.csv")),15)
            self.assertEqual(len(read(p/"out"/"table9_mrpc.csv")),5)
            self.assertFalse((p/"out"/"mrpc_ablation.csv").exists())
            self.assertTrue(all(r["mean"]=="尚未完成" for r in read(p/"out"/"task_summary.csv")))
