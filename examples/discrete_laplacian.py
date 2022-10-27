import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import torch
import torch.nn as nn

sys.path.append('../')
from torchdiffeq import odeint as odeint_default
from proxNode.lib.prox.odeprox import odeint
from tqdm import tqdm

DEVICE='cpu'
torch.device(DEVICE)
torch.random.manual_seed(0)

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
ODE Solver
    - 
"""
class solver(nn.Module):
    def __init__(self, func, opt, **kwargs):
        super(solver, self).__init__()
        self.func = func
        self.opt = opt
        
    def forward(self, t, x):
        opt=self.opt
        if opt['prox_method'] in ('dopri5', 'adaptive_heun'):
            out = odeint_default(self.func, x, t, method=opt['prox_method'],rtol=opt['rtol'],atol=opt['atol'])
        else:
            out = odeint(self.func, x, t, opt_method=opt['opt_method'],prox_method=opt['prox_method'],
                        int_step=opt['int_step'],opt_step=opt['opt_step'],
                        options=self.opt)
        return out

    def _get_nfes(self):
        return self.func.nfe
    def _set_nfes(self, val):
        self.func.nfe = val
        return 1


""" Explicit Routine """
def explicit(func,h,t,method,rtols,atols):

    dataframe = pd.DataFrame()

    for rtol,atol in tqdm(zip(rtols,atols),total=len(rtols),desc=method):
        
        options = {'prox_method':method,'rtol':rtol,'atol':atol}

        sol = solver(func,options)
        sol._set_nfes(0)

        appx = sol.forward(t,h)[-1]
        err = torch.norm(appx-true).cpu().detach().float()
        rel_err = err/torch.norm(true).cpu().detach().float()

        dataframe = dataframe.append({'Method':method,'Step':0,'NFE':sol._get_nfes(),'Error':float(err),'Rel_Err':float(rel_err)},ignore_index=True)

    return dataframe

""" Proximal Routine """
def proximal(func,h,t,opt_method,prox_method,int_steps,opt_steps,opt_iters,opt_tols):

    dataframe = pd.DataFrame()

    for int_step in tqdm(int_steps,total=len(int_steps)):
        for opt_step in opt_steps:
        
            options = {'opt_method':opt_method,'prox_method':prox_method,
                    'int_step':int_step,'opt_step':opt_step,
                    'opt_iters':100,'grad_tol':1e-5, 'opt_tol':1e-6,
                    'device':DEVICE}

            sol = solver(func,options)
            sol._set_nfes(0)

            appx = sol.forward(t,h)[-1]
            err = torch.norm(appx-true).detach().float()
            rel_err = err/torch.norm(true).detach().float()

            dataframe = dataframe.append({'Method':prox_method,'Step':int_step,'NFE':sol._get_nfes(),'Error':float(err),'Rel_Err':float(rel_err)},ignore_index=True)

    return dataframe

""" Fast Fourier Transform Routine """
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


""" Grid Configuration """

n = 128
itv = [0, 1]
t = 1
h0 = torch.randn(n)

# Prep
dx = (itv[1]-itv[0])/n
xrange = torch.arange(itv[0], itv[1], dx)
h0 = torch.nn.Parameter(h0)


true = fft_solve(h0,dx,t).to(DEVICE)
t = torch.Tensor([0,t]).to(DEVICE)
h0=h0.to(DEVICE)


"""
PROXIMAL METHOD EXPERIMENTS 

"""
N=11
func = ode(dx)
df = pd.DataFrame(columns={'Method','NFE','Error','Rel_Err'})

"DOPRI5"
rtols=np.logspace(-1,-6,N)
atols=[1e-5]*N
df_temp = explicit(func,h0,t,'dopri5',rtols,atols)
df = df.append(df_temp,ignore_index=True)

"Adaptive Heun"
rtols=np.logspace(-1,-6,N)
atols=[1e-5]*N
df_temp = explicit(func,h0,t,'adaptive_heun',rtols,atols)
df = df.append(df_temp,ignore_index=True)

"CRANK NICOLSON"
int_steps = [1/2000]*N
opt_steps = [.1]*N
opt_tols = np.logspace(-3,-7,N)
opt_iters = [100]*N
df_temp = proximal(func,h0,t,'fletch_reeves','crank_nicolson',int_steps,opt_steps,opt_iters,opt_tols)
df = df.append(df_temp,ignore_index=True)

"BDF2"
int_steps = [1/2000]*N
opt_steps = [.1]*N
opt_tols = np.logspace(-3,-7,N)
opt_iters = [100]*N
df_temp = proximal(func,h0,t,'fletch_reeves','bdf2',int_steps,opt_steps,opt_iters,opt_tols)
df = df.append(df_temp,ignore_index=True)

"BDF3"
int_steps = [1/2000]*N
opt_steps = [.1]*N
opt_tols = np.logspace(-3,-7,N)
opt_iters = [100]*N
df_temp = proximal(func,h0,t,'fletch_reeves','bdf3',int_steps,opt_steps,opt_iters,opt_tols)
df = df.append(df_temp,ignore_index=True)

"BDF4"
int_steps = [1/2000]*N
opt_steps = [.1]*N
opt_tols = np.logspace(-3,-7,N)
opt_iters = [100]*N
df_temp = proximal(func,h0,t,'fletch_reeves','bdf4',int_steps,opt_steps,opt_iters,opt_tols)
df = df.append(df_temp,ignore_index=True)

"EULER1"
int_steps = [1/2000]*N
opt_steps = [.1]*N
opt_tols = np.logspace(-3,-7,N)
opt_iters = [100]*N
df_temp = proximal(func,h0,t,'fletch_reeves','euler',int_steps,opt_steps,opt_iters,opt_tols)
df = df.append(df_temp,ignore_index=True)

"Output"
df.to_csv('./out/discrete_laplacian.csv')