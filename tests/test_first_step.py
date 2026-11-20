import unittest
import torch
from glue.first_step import cosine,direction


class DirectionTests(unittest.TestCase):
    def test_signed_derivative_and_quadratic_term(self):
        gen=torch.Generator().manual_seed(23)
        def matrix(n,m):
            return torch.randn(n,m,generator=gen,dtype=torch.float64,requires_grad=True)
        for zero_b in (False,True):
            a,b=matrix(2,5),matrix(7,2)
            if zero_b:
                b=torch.zeros_like(b,requires_grad=True)
            g=matrix(7,5).detach()
            for s in (2.,16/8**.5):
                da,db=torch.autograd.grad((s*(b@a)*g).sum(),(a,b))
                update=-s*(db@a+b@da)
                torch.testing.assert_close(direction(g,a,b,s),update)
                eta=1e-4
                finite=(s*(b-eta*db)@(a-eta*da)-s*b@a)/eta
                torch.testing.assert_close(finite-eta*s*db@da,update)
                self.assertGreater(cosine(update,-g),0)
        self.assertIsNone(cosine(torch.zeros_like(g),g))
        self.assertFalse(torch.cuda.is_initialized())
