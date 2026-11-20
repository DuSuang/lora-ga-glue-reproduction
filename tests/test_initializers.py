import math
import unittest
import torch
from glue.adapters import Adapter, METHODS, initialize_factors, resolved_init, scale


class InitializerTests(unittest.TestCase):
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
