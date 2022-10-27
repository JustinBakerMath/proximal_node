import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import fsolve
import sys

parser = argparse.ArgumentParser(description='Test Equation Parameters.')
parser.add_argument('-k', metavar='ode_coeff', type=int, default=3,
                    help='k<0: well-posed + stiff | k>0: ill-posed + non-stiff')
parser.add_argument('-T', metavar='terminal_time', type=int, default=1,
                    help='note: internal dt=.01, generally choose T>1')
parser.add_argument('-vis', type=bool, default=False,
                    help='visualize the data after code execution')
args = parser.parse_args()

k=args.k
""" y'=ky+(1-k)cos(t)-(1+k)sin(t)"""
def ode(t,y):
  return k*y-(1+k)*np.sin(t)+(1-k)*np.cos(t)
def true(t):
  return np.cos(t)+np.sin(t)

def integrate(y0,t0,tn,dt,ode,ds,step):
  t = np.linspace(t0,tn,int((tn-t0)/dt+1))
  yk = y0
  for t_ in t:
    yk = step(yk,t_,dt,ode,ds)
  return yk


"""
Proximal Block
    - Solve proximal using backward euler scheme and gradient descent
    - grad_desc: standard gradient descent
    - proximal: backward euler scheme for proximal algorithm
    - prox_step: solves the proximal algorithm
"""
def grad_desc(dF,t,ds):
  return t-ds*dF
def proximal(z0,zn,tn,dt,ode):
  return zn-z0-dt*ode(tn,zn)
def prox_step(hk,tk,dt,ode,ds,opt_tol=1e-10):
  i=0
  z0=hk
  zn=z0
  converged=False
  while (i<100) and not converged: #tunable max iterations
    dP = proximal(z0,zn,tk,dt,ode)
    zn1 = grad_desc(dP,zn,ds)
    if abs(dP)<opt_tol: #tunable stopping criteria
      converged = True
    zn = zn1
  return zn

""" Explicit Euler Block """
def explicit_step(hk,tk,dt,ode,ds):
  return hk+dt*ode(tk,hk)

"""
Implicit Euler Block
    - Solve implicit scheme using residuals
    - implicit_step: approximates next value,
                     then solves the linear equation
                     for the residual
    - residual: computes the residual for the implicit euler scheme
"""
def residual(hk,ode,ti,hi,tk):
    return hk-hi-(tk-ti)*ode(ti,hk)
def implicit_step(hk,tk,dt,ode,ds):
    hk_est = hk + dt*ode(tk,hk)
    hk = fsolve(residual,hk_est,args=(ode,tk,hk,tk+dt))[0]
    return hk


""" Compare stepwise error """
T=args.T
dt=.01
ds=.01
y0=true(0)
opt_tols = [1e-4,1e-5,1e-6]
df = pd.DataFrame(columns={'Method','Time','Error'})

"""Explicit"""
t = np.linspace(0,T,int(T/dt+1))
yk = y0
for t_ in t[1:]:
    yk = explicit_step(yk,t_,dt,ode,ds)
    err = abs(true(t_)-yk)
    rel_err = err/abs(true(t_))
    df=df.append({'Method':'Forward Euler','Time':t_,'Error':err,'Rel Err':rel_err},ignore_index=True)

"""Proximal"""
for opt_tol in opt_tols:
    t = np.linspace(0,T,int(T/dt+1))
    yk = y0
    for t_ in t[1:]:
        yk = prox_step(yk,t_,dt,ode,ds,opt_tol=opt_tol)
        err = abs(true(t_)-yk)
        rel_err = err/abs(true(t_))
        df=df.append({'Method':'Prox {:.0E}'.format(opt_tol),'Time':t_,'Error':err,'Rel Err':rel_err},ignore_index=True)

"""Implicit"""
t = np.linspace(0,T,int(T/dt+1))
yk = y0
for t_ in t[1:]:
    yk = implicit_step(yk,t_,dt,ode,ds)
    err = abs(true(t_)-yk)
    rel_err = err/abs(true(t_))
    df=df.append({'Method':'Backward Euler','Time':t_,'Error':err,'Rel Err':rel_err},ignore_index=True)

"""
Output
    - Save Dataframe
    - Generate Error Plot
    - Generate Relative Error Plot
"""
df.to_csv('./out/test_eqn.csv')