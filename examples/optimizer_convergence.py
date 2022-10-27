import abc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


"""
ODE
    - dh/dt=-L*h(t,x)
    - L: discrete laplacian
"""
class ode(nn.Module):
    def __init__(self,dx):
        super().__init__()
        self.dx = dx
        self.c = 1/dx/dx
        self.nfe = 0
    def forward(self,t,x):
        self.nfe += 1
        return self.c*(torch.roll(x,1)+torch.roll(x,-1)-2*x)
"""
Fast Fourier Transform Routine

"""
def fft_solve(y0,dx,t):
    n=len(y0)
    # Coefficients
    ctemp = torch.zeros(n)
    ctemp[-1] = 1
    ctemp[1] = 1
    ctemp[0] = -2
    fftc = torch.fft.fft(ctemp) / dx / dx
    # Transform
    w_0 = torch.fft.fft(y0)
    w_t = torch.exp(t * fftc) * w_0
    u_t = torch.real(torch.fft.ifft(w_t))

    return u_t


"""
Proximal Methods
"""
class proximal(metaclass=abc.ABCMeta):
  def __init__(self):
    self.grad = None

class euler(proximal):
  def step_grad(self,dF,dt,tn,hk,zn):
    self.grad=zn-hk-dt*dF(tn,zn)
    return self.grad

PROXIMALS ={'euler':euler()}

"""
Inner Optimization Methods

"""
class optimizer(metaclass=abc.ABCMeta):
  def __init__(self,proximal):
    self.prox = proximal

class grad_desc(optimizer):
    def iterate(self, itr, ds, zn, dProx, hk,dt,ode):
        return zn-ds*dProx

class nesterov(optimizer):
    def iterate(self, itr, ds, zn, dProx, hk,dt,ode):
        xk1 = zn - ds*dProx
        if itr ==0:
            self.xk = xk1
        zn = xk1+(itr/(itr+3))*(xk1-self.xk)
        self.xk=xk1
        self.zn = zn
        return zn

class nesterov_restart(optimizer):
    def __init__(self,proximal):
        super(nesterov_restart,self).__init__(proximal)
        self.k=0

    def iterate(self, itr, ds, zn, dProx, hk,dt,ode):

        xk1 = zn - ds*dProx
        if itr==0:
            self.xk = xk1

        zn1 = xk1+(self.k)/(self.k+3)*(xk1-self.xk)

        self.k=self.k+1
        if torch.norm(dProx@zn)<torch.norm(dProx@zn1):
            self.k=0
        self.xk=xk1
        zn = zn1
        return zn

class fletch_reeves(optimizer):
    def iterate(self,itr,ds,zn,dProx,hk,dt,dF):
        if itr ==0:
            self.btol=torch.Tensor([1e-6])
            self.beta=torch.Tensor([1-self.btol])
            self.grad_zn = dProx
            self.pk = torch.zeros_like(self.grad_zn)
        pk1 = self.beta*self.pk + self.grad_zn
        zn1 = zn-ds*pk1
        grad_zn1 = zn1-hk-dt*dF(dt,zn1)

        numerator = torch.tensordot(grad_zn1,grad_zn1,dims=len(grad_zn1.shape))
        denominator = torch.tensordot(self.grad_zn,self.grad_zn,dims=len(self.grad_zn.shape))
 
        beta_cand = torch.Tensor([1-self.btol,numerator/denominator])
        self.beta = torch.min(beta_cand).float()
        self.pk = pk1
        self.grad_zn = grad_zn1
        zn=zn1
        return zn

class lbfgs(optimizer):
    def iterate(self,itr,ds,zn,dProx,hk,dt,dF):
        if itr==0:
            self.I = torch.eye(zn.shape[-1])
            self.Bi = torch.eye(zn.shape[-1])
            self.grad_zn = dProx 
            return zn-ds*self.grad_zn
      
        zn1 = zn - ds*(self.Bi@self.grad_zn)
        self.grad_zn1 = zn1-hk-dt*dF(dt,zn1)

        self.v = zn1-zn
        self.r = self.grad_zn1-self.grad_zn
        if len(self.v.shape)<2:
            self.v = self.v[None,:]
        if len(self.r.shape)<2:
            self.r = self.r[None,:]
        inner_prod = torch.einsum('ij,ij->i',self.v,self.r)

        if not torch.all(inner_prod>0):
            self.grad_zn=dProx
            return zn1

        self.rho = torch.pow(inner_prod,-1)

        outer_prod = torch.einsum('ij,ik->ijk',self.r,self.v)
        Z_ = torch.einsum('i,ijk->jk',self.rho,outer_prod)
        self.Z = self.I-Z_

        outer_prod = torch.einsum('ij,ik->ijk',self.v,self.v)
        Bi_ = torch.einsum('i,ijk->jk',self.rho,outer_prod)

        self.Bi = (self.Z.T)@self.Bi@(self.Z)+Bi_

        zn = zn1
        self.grad_zn=self.grad_zn1
        return zn

OPTIMIZERS={'grad_desc':grad_desc, 'nesterov':nesterov, 'nesterov_restart':nesterov_restart, 'fletch_reeves':fletch_reeves, 'lbfgs':lbfgs}

"""
Measure of Convergence
    - several different measures of err (i.e. convergence) may be selected via commenting 179-181
"""

def convergence(true,z0,dt,ds,ode,prox_str,opt_str,p=2,tol=1e-16):
  prox = PROXIMALS[prox_str]
  opt = OPTIMIZERS[opt_str](prox)
  dataframe = pd.DataFrame(columns={'Iteration','err','Method'})
  zn = z0
  zn1=zn
  i=0
  converged = False
  while (i<1000) and not converged:
    zn = zn1
    dProx = prox.step_grad(ode,dt,dt,z0,zn)
    zn1 = opt.iterate(i,ds,zn,dProx,z0,dt,ode)
    err = torch.norm(zn1-zn).detach().float()
#    err = (torch.norm(ode(dt,zn1))/torch.norm(ode(dt,z0))).detach().float()
#    err = torch.norm(ode(dt,zn1)).detach().float()
    # err = torch.norm(dProx/ds).detach().float()
    dataframe = dataframe.append({'Iteration':i,'err':float(err),'Method':opt_str},ignore_index=True)
    i+=1
    if torch.norm(zn-zn1)<tol:
        converged=False
  return dataframe

""" Grid """
n = 128
itv = [0, 1]
t = 1
h0 = torch.randn(n)

# Initialize
dx = (itv[1]-itv[0])/n
xrange = torch.arange(itv[0], itv[1], dx)

h0 = torch.nn.Parameter(h0)
h0_np = h0.detach().numpy()
true = fft_solve(h0,dx,t)
t = torch.Tensor([0,t])

""" Execution """
df = pd.DataFrame()
df_temp = convergence(true,h0,.01,.001,ode(dx),'euler','grad_desc',p=2)
df = df.append(df_temp,ignore_index=True)

df_temp = convergence(true,h0,.01,.001,ode(dx),'euler','nesterov',p=2)
df = df.append(df_temp,ignore_index=True)

df_temp = convergence(true,h0,.01,.001,ode(dx),'euler','fletch_reeves',p=2)
df = df.append(df_temp,ignore_index=True)

df_temp = convergence(true,h0,.01,.001,ode(dx),'euler','nesterov_restart',p=2)
df = df.append(df_temp,ignore_index=True)

df_temp = convergence(true,h0,.01,.001,ode(dx),'euler','lbfgs',p=2)
df = df.append(df_temp,ignore_index=True)

"Results"
df.to_csv('./out/optimizer_convergence.csv')