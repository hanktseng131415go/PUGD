# -*- coding: utf-8 -*-
"""
PUGD in Pytorch
by Ching-Hsun Tseng
"""
import torch

class PUGD(torch.optim.Optimizer):
    
    def __init__(self, params, base_optimizer, **kwargs):
        
        defaults = dict(**kwargs)
        super(PUGD, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)
        
    @torch.no_grad()
    def xp_step(self, zero_grad=True):  
        '''UGD = NGD-FW in Tensor'''
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
        
                p.grad = p.grad / (grad_norm + 1e-12)
                
        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):

        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:
            
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue

                self.state[i]["e_w"] = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])
        
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
            
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()
        
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
    
    def _abs_grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
