import math
import unittest
import torch
from glue.adapters import Adapter, METHODS, initialize_factors, resolved_init, scale
from glue.glue_train import GA_INIT_LOGIT_RTOL, INIT_LOGIT_ATOL, INIT_LOGIT_RTOL, check_initial_logits


class InitializerTests(unittest.TestCase):
    def test_fp32_initial_probe(self):
        before = torch.tensor([[[1., 1.000001, 23.]]], dtype=torch.float32)
        after = before.clone()
        after[0, 0, 2] += 0.0004673
        with self.assertRaises(AssertionError):
            check_initial_logits(before, after, [0, 1], rtol=INIT_LOGIT_RTOL, atol=INIT_LOGIT_ATOL)
        check_initial_logits(before, after, [0, 1], rtol=GA_INIT_LOGIT_RTOL, atol=INIT_LOGIT_ATOL)
        with self.assertRaises(AssertionError):
            check_initial_logits(before, after, [1, 2], rtol=GA_INIT_LOGIT_RTOL, atol=INIT_LOGIT_ATOL)
        invalid = {
            "large vocabulary drift": (2, 23.01),
            "changed class": (0, 1.000002),
            "tied classes": (0, before[0, 0, 1].item()),
            "non-finite score": (2, float("nan")),
        }
        for case, (index, value) in invalid.items():
            changed = before.clone()
            changed[0, 0, index] = value
            with self.subTest(case=case), self.assertRaises(AssertionError):
                check_initial_logits(before, changed, [0, 1], rtol=GA_INIT_LOGIT_RTOL, atol=INIT_LOGIT_ATOL)

    def test_five_branches_and_compensation(self):
        grad = torch.diag(torch.arange(48.,0.,-1.)).double()[:40]
        results={}
        for method in METHODS:
            torch.default_generator.manual_seed(144)
            base=torch.nn.Linear(48,40,bias=False,dtype=torch.float64)
            original=base.weight.detach().clone()
            module=Adapter(base,scale(method))
            initialize_factors(module,method,grad,svd_seed=12)
            results[method]=(module.A.clone(),module.B.clone())
            module.compensate()
            torch.testing.assert_close(module.weight+module.factor*module.B@module.A,original)
            self.assertEqual(module.factor,resolved_init(method)["scaling"])
            self.assertEqual(module.B.norm().item()==0,method=="lora")
        c=40**0.25/4
        for raw,stable in [("gaussian","so"),("ga_only","ga")]:
            for x,y in zip(results[raw],results[stable]):
                torch.testing.assert_close(x*c,y)
        a,b=results["ga_only"]
        self.assertGreater(a[:,:8].square().sum().item(),7.9)
        self.assertGreater(b[8:16].square().sum().item(),7.9)
        self.assertEqual([scale(m) for m in METHODS],[2,2,16/math.sqrt(8),2,16/math.sqrt(8)])
        self.assertFalse(torch.cuda.is_initialized())
