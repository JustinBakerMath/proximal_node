import abc
import torch
from .proximals import * 
import numpy as np

#TODO: Larger stepsizes in experiments

class Optimizer(metaclass=abc.ABCMeta):
    order:int
    def __init__(self, proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, **unused_kwargs):

        self.proximal = proximal
        self.step_size = step_size
        self.tol = tol
        self.max_itrs = max_itrs
        self.is_multistep = is_multistep
        self.device = device

        self.zn = None
        self.halt = False

    def minimize(self, dt, tk, hk):
        zn = hk
        tn = tk
        itr=0
        for itr in range(self.max_itrs):
            zn1 = self.iterate(itr, dt, tk, tn, hk, zn)
            self.halt = self.proximal.check_halt()
            if (torch.norm(zn-zn1) < self.tol) or self.halt:
                break
            else:
                zn = zn1
        return zn

"""
Identity Optimizer: returns zn-dP'

"""
class IdentityOptimizer(Optimizer):
    def minimize(self, dt, tk, hk):
        zn = hk
        return self.iterate(torch.zeros_like(zn), dt, tk, tk, hk, hk)
    def iterate(self, itr, dt, tk, tn, hk, zn):
        zn = -self.proximal.gradient(itr, dt, tk, tn, hk, zn)+zn
        self.zn = zn
        return zn

"""
Gradient Descent:
    minimizes: zn1=zn-s*dP'
    s: opt_step
"""
class grad_desc(Optimizer):
    def iterate(self, itr, dt, tk, tn, hk, zn):
        gradProx = self.proximal.gradient(itr, dt, tk, tn, hk, zn)
        zn = zn-self.step_size*gradProx
        self.zn = zn
        return zn

"""
Fletcher Reeves Accelerated Gradient Descent:
    minimizes:
"""
class grad_desc_fr(Optimizer):
    def __init__(self, proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, btol=None, **unused_kwargs):
        super(grad_desc_fr, self).__init__( proximal, step_size, tol, max_itrs, device, is_multistep)
        if btol is None:
            btol = torch.Tensor([1e-6]).to(self.device)
        self.btol = torch.Tensor([btol]).to(self.device)
        self.beta = torch.Tensor([1-btol]).to(self.device)
        self.pk = torch.Tensor([]).to(self.device)

    def iterate(self, itr, dt, tk, tn, hk, zn):
        if itr ==0:
            self.grad_zn = self.proximal.gradient(itr, dt, tk, tn, hk, zn)
            self.pk = torch.zeros_like(self.grad_zn).to(self.device)
        pk1 = self.beta*self.pk + self.grad_zn
        zn1 = zn-self.step_size*pk1
        grad_zn1 = self.proximal.gradient(itr, dt, tk, tn, hk, zn1)

        numerator = torch.tensordot(grad_zn1,grad_zn1,dims=len(grad_zn1.shape))
        denominator = torch.tensordot(self.grad_zn,self.grad_zn,dims=len(self.grad_zn.shape))
 
        beta_cand = torch.Tensor([1-self.btol,numerator/denominator])
        self.beta = torch.min(beta_cand).float()
        self.pk = pk1
        self.grad_zn = grad_zn1
        zn=zn1

        return zn

"""
Nesterov Accelerated Gradient Descent:
    minimizes:
"""
class nesterov(Optimizer):
    def __init__(self, proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, **unused_kwargs):
        super(nesterov, self).__init__( proximal, step_size, tol, max_itrs, device, is_multistep)

    def iterate(self, itr, dt, tk, tn, hk, zn):
        xk1 = zn - self.step_size*self.proximal.gradient(itr, dt, tk, tn, hk, zn)
        if itr ==0:
            self.xk = xk1
        zn = xk1+(itr/(itr+3))*(xk1-self.xk)
        self.xk=xk1
        self.zn = zn
        return zn

"""
Nesterov Accelerated Gradient Descent with Restarting:
    minimizes:
"""
class nesterov_restart(Optimizer):
    def __init__(self,  proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, restart_itrs=None, **unused_kwargs):
        super(nesterov_restart, self).__init__( proximal, step_size, tol, max_itrs, device, is_multistep)
        if restart_itrs is None:
            restart_itrs = 20
        self.restart_itrs = restart_itrs
    def minimize(self, dt, tk, hk):
        zn = hk
        tn = tk
        itr=0
        for itr in range(self.max_itrs):
            i = itr%self.restart_itrs
            zn1 = self.iterate(i, dt, tk, tn, hk, zn)
            self.halt = self.proximal.check_halt()
            if (torch.norm(zn-zn1) < self.tol) or self.halt:
                break
            else:
                zn = zn1
        return zn
    def iterate(self, itr, dt, tk, tn, hk, zn):
        xk1 = zn - self.step_size*self.proximal.gradient(itr, dt, tk, tn, hk, zn)
        if itr ==0:
            self.xk = xk1
        zn = xk1+(itr/(itr+3))*(xk1-self.xk)
        self.xk=xk1
        self.zn = zn
        return zn

"""
Barzilai-Borwein Step Method:
    minimizes:
"""
class bbstep(Optimizer):
    def __init__(self,  proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, alpha_default=None, **unused_kwargs):
        super(bbstep, self).__init__( proximal, step_size, tol, max_itrs, device, is_multistep)
        self.grad_zn = None
        self.s = None #v > s
        self.y = None# r>y
        self.alpha1 = 1
        self.alpha2 = 1
        if alpha_default is None:
            self.alpha_default = 1e-3
        self.alphainv = self.alpha_default
        
    def iterate(self, itr, dt, tk, tn, hk, zn):
        if itr==0:
            self.I = torch.eye(zn.shape[-1])
            self.grad_zn = self.proximal.gradient(itr, dt, tk, tn, hk, zn)
        zn1 = zn - self.step_size*(self.alphainv*self.grad_zn.T).T
        self.grad_zn1 = self.proximal.gradient(itr, dt, tk, tn, hk, zn1)
        self.s = zn1-zn
        self.y = self.grad_zn1-self.grad_zn
        if len(self.s.shape)<2:
            self.s = self.s[None,:]
            self.y = self.y[None,:]
        numerator = torch.einsum('ij,ij->i',self.s,self.y)
        denominator = torch.einsum('ij,ij->i',self.s,self.s)
        self.alpha1 = numerator/denominator
        numerator = torch.einsum('ij,ij->i',self.y,self.y)
        denominator = torch.einsum('ij,ij->i',self.s,self.y)
        self.alpha2 = numerator/denominator       
        if all(self.alpha1>1e9):
            self.alphainv = 1/self.alpha1
        elif all(self.alpha2>1e9):
            self.alphainv = 1/self.alpha2
        else:
            self.alphainv = self.alpha_default
        zn = zn1
        self.grad_zn=self.grad_zn1
        self.zn = zn
        return zn

"""
Broyden-Fletcher-Goldfarb-Shanno (BFGS):
    minimizes:
"""
class lbfgs(Optimizer):
    def __init__(self,  proximal, step_size, tol, max_itrs, device='cpu', is_multistep=False, **unused_kwargs):
        super(lbfgs, self).__init__( proximal, step_size, tol, max_itrs, device, is_multistep)
        self.Bi = None
        self.grad_zn = None
        self.v = None
        self.r = None
        self.rho = None
        self.Z = None
        
    def iterate(self, itr, dt, tk, tn, hk, zn):
        if itr==0:
            self.I = torch.eye(zn.shape[-1]).to(self.device)
            self.Bi = torch.eye(zn.shape[-1]).to(self.device)
            self.grad_zn = self.proximal.gradient(itr, dt, tk, tn, hk, zn)
        zn1 = zn - self.step_size*(self.Bi@self.grad_zn.T).T
        self.grad_zn1 = self.proximal.gradient(itr, dt, tk, tn, hk, zn1)
        self.v = zn1-zn
        self.r = self.grad_zn1-self.grad_zn
        if len(self.v.shape)<2:
            self.v = self.v[None,:]
        if len(self.r.shape)<2:
            self.r = self.r[None,:]
        inner_prod = torch.einsum('ij,ij->i',self.v,self.r)
        self.rho = torch.pow(inner_prod,-1)
        outer_prod = torch.einsum('ij,ik->ijk',self.r,self.v)
        Z_ = torch.einsum('i,ijk->jk',self.rho,outer_prod)
        self.Z = self.I-Z_
        outer_prod = torch.einsum('ij,ik->ijk',self.v,self.v)
        Bi_ = torch.einsum('i,ijk->jk',self.rho,outer_prod)
        self.Bi = self.Z@self.Bi@self.Z.T+Bi_
        zn = zn1
        self.grad_zn=self.grad_zn1
        return zn