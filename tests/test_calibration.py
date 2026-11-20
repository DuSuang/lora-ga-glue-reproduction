import unittest
from types import SimpleNamespace

import torch

from glue.adapters import estimate
from glue.first_step import gradients


class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[1.,2.]]))

    def forward(self,x,labels):
        return SimpleNamespace(loss=(self.proj(x).flatten()-labels).square().mean())


class CalibrationTests(unittest.TestCase):
    def test_singleton_mean(self):
        for lengths in [(1,3),(2,2)]:
            def batch(indices,initializing=False):
                i,=indices
                x=torch.eye(2,dtype=torch.float64)[i].repeat(lengths[i],1)
                return dict(x=x,labels=torch.zeros(lengths[i],dtype=torch.float64))
            data=SimpleNamespace(batch=batch)
            model=Example()
            gs=[torch.autograd.grad(model(**batch([i])).loss,model.proj.weight)[0] for i in range(2)]
            equal=(gs[0]+gs[1])/2
            token=(lengths[0]*gs[0]+lengths[1]*gs[1])/sum(lengths)
            measured=estimate(model,{"proj":model.proj},data,dropout=False,indices=range(2))["proj"]
            torch.testing.assert_close(measured,equal,atol=0,rtol=0)
            model.proj.weight.requires_grad_(True)
            diagnostic,=gradients(model,data,range(2),[model.proj.weight],True)
            torch.testing.assert_close(diagnostic,equal,atol=0,rtol=0)
            self.assertEqual((equal-token).abs().max().item(),1. if lengths==(1,3) else 0.)
