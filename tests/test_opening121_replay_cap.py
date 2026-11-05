import copy,unittest
import torch
from cap_opening121_replay import cap_coefficient,policy_gradients,norm_of,dot_of
from az121 import Net121
from ablate_opening121_hard32 import terms
from opening121 import OpeningBook
from train_opening121 import book_samples
from opening121_ratios import mixed_loss

class ReplayCapTests(unittest.TestCase):
    def test_cap_edge_cases_and_detachment(self):
        for root,replay,want in [(2.,4.,.5),(4.,2.,1.),(0.,4.,0.),(0.,0.,1.)]:
            weight=cap_coefficient(torch.tensor(root,requires_grad=True),torch.tensor(replay,requires_grad=True));self.assertAlmostEqual(float(weight),want);self.assertFalse(weight.requires_grad)
        self.assertEqual(float(cap_coefficient(torch.tensor(1.),torch.tensor(10.),False)),1.)

    def test_nonfinite_norm_rejected(self):
        for value in (float('inf'),float('nan')):
            with self.assertRaises(ValueError):cap_coefficient(torch.tensor(value),torch.tensor(1.))
            with self.assertRaises(ValueError):cap_coefficient(torch.tensor(1.),torch.tensor(value))

    def test_control_all_gradients_match_and_value_head_is_aux_only(self):
        torch.set_num_threads(1);m=Net121(width=4,canonical=True);reference=copy.deepcopy(m)
        with torch.no_grad():
            for p in reference.parameters():p.add_(.01)
        sample=book_samples(OpeningBook())[0]['train'][0];roots=[{**sample,'value_weight':0.}]*4;ordinary=[{**sample,'value_weight':1.,'z':1.}]*12
        root,replay,aux=terms(m,reference,roots,ordinary);named=list(m.named_parameters());params=[p for _,p in named];shared=[p for n,p in named if not n.startswith('value.')]
        rg=policy_gradients(root,shared);pg=policy_gradients(replay,shared);w=cap_coefficient(norm_of(rg),norm_of(pg),False)
        actual=torch.autograd.grad(root+w*replay+aux,params,retain_graph=True);expected=torch.autograd.grad(mixed_loss(m,reference,roots,ordinary),params,retain_graph=True)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        value_params=[p for n,p in named if n.startswith('value.')];aux_grads=torch.autograd.grad(aux,value_params,retain_graph=True)
        capped=cap_coefficient(norm_of(rg),norm_of(pg));all_grads=torch.autograd.grad(root+capped*replay+aux,value_params)
        for a,b in zip(all_grads,aux_grads):torch.testing.assert_close(a,b)

    def test_weighted_gradient_cap_does_not_scale_other_targets(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));unused=torch.nn.Parameter(torch.tensor([3.]));root=(p*torch.tensor([.25,.5])).sum();replay=(p*torch.tensor([-3.,4.])).sum();aux=p.square().sum()
        gr=policy_gradients(root,[p,unused]);gp=policy_gradients(replay,[p,unused]);weight=cap_coefficient(norm_of(gr),norm_of(gp));loss=root+weight*replay+aux
        self.assertIsNone(p.grad);self.assertIsNone(unused.grad);self.assertEqual(float(gr[1].abs().sum()),0.)
        loss.backward();torch.testing.assert_close(p.grad,torch.tensor([.25,.5])+weight*torch.tensor([-3.,4.])+2*p.detach());self.assertLessEqual(float(norm_of(tuple(weight*g for g in gp))),float(norm_of(gr))+1e-6)
        self.assertAlmostEqual(float(dot_of(gr,gp)),1.25)

if __name__=='__main__':unittest.main()
