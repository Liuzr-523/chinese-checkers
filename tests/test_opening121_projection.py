import unittest
import torch
from project_opening121_replay import projection_coefficient,projected_objective,dot_of

class ProjectionTests(unittest.TestCase):
    def test_conflict_and_nonconflict(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([-2.,3.]);c=projection_coefficient((a*b).sum(),a.square().sum());torch.testing.assert_close(b+c*a,torch.tensor([0.,3.]));self.assertFalse(c.requires_grad)
        self.assertEqual(float(projection_coefficient(torch.tensor(2.),torch.tensor(1.))),0.)
        self.assertEqual(float(projection_coefficient(torch.tensor(-2.),torch.tensor(0.))),0.)
        self.assertEqual(float(projection_coefficient(torch.tensor(-2.),torch.tensor(1.),False)),0.)

    def test_exact_gradient_with_aux_and_value_independent(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));value=torch.nn.Parameter(torch.tensor(3.));root=p[0];replay=-2*p[0]+3*p[1];aux=p.square().sum()+value.square()
        loss,a,b,c=projected_objective(root,replay,aux,[p]);self.assertIsNone(p.grad);self.assertGreaterEqual(float(dot_of(a,tuple(g+c*r for g,r in zip(b,a)))),0.)
        loss.backward();torch.testing.assert_close(p.grad,torch.tensor([1.,3.])+2*p.detach());self.assertEqual(float(value.grad),6.)

    def test_invalid_numbers_rejected(self):
        for dot,norm in [(float('inf'),1.),(0.,float('nan')),(1.,-1.)]:
            with self.assertRaises(ValueError):projection_coefficient(torch.tensor(dot),torch.tensor(norm))

if __name__=='__main__':unittest.main()
