import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

sys.path.append('../')

plt.style.use('classic')
sns.set()

"""
Example 1: 
    -ode: h'=kh+(1-k)cos(t)-(1+k)sin(t), k=10, t=[0,1]
"""
# Load data
prox = pd.read_csv('./out/test_eqn.csv')

opt_tols = [1e-4,1e-5,1e-6]

g=sns.relplot(data=prox[prox['Method']!='Backward Euler'],
                x='Time', y='Error',
                hue='Method',kind='line',
                palette=['k','tab:orange','tab:red','tab:blue'],
                legend=False)
# g.set(yscale='log')
labels = ['Forward Euler']+['Prox'+'%.0E'%opt_tol for opt_tol in opt_tols]
plt.legend(labels=labels)
plt.savefig('./out/test_eqn_fwd.pdf')

g=sns.relplot(data=prox[prox['Method']!='Forward Euler'],
                x='Time', y='Error',
                hue='Method',kind='line',
                palette=['tab:orange','tab:red','tab:blue','k'],
                legend=False)
labels = ['Prox'+'%.0E'%opt_tol for opt_tol in opt_tols]+['Backward Euler']
plt.legend(labels=labels)
plt.savefig('./out/test_eqn_bkwd.pdf')


"""
Example 2(NFE): 
    -ode: dh/dt=-L*h(t,x)
    - L: discrete laplacian
"""

df = pd.read_csv('./out/discrete_laplacian_prox.csv')
plt.style.use('classic')
sns.set()
plt.figure(figsize=(18, 10))
axes = plt.gca()
axes.set_ylim([9.5*1000,23*10000])
axes.set_xlim([0.000001,0.15])
axes.set_xscale('log')
axes.set_yscale('log')

axes.tick_params(axis='x', labelsize=35+10)
axes.tick_params(axis='y', labelsize=35+10)

dopri = df[df['Method']=='dopri5']
ah = df[df['Method']=='adaptive_heun']
cn = df[df['Method']=='crank_nicolson']
bdf2 = df[df['Method']=='bdf2']
bdf3 = df[df['Method']=='bdf3']
bdf4 = df[df['Method']=='bdf4']
euler = df[df['Method']=='euler']
euler3 = df[df['Method']=='euler3']
plt.plot(dopri['Error'], dopri['NFE'], 'o', markersize=20, color='tab:orange', label = "DOPRI5")
plt.plot(ah['Error'], dopri['NFE'], 'o', markersize=20, color='tab:red', label = "Adaptive Heun")
plt.plot(bdf2['Error'], bdf2['NFE'], 'o', markersize=20, color='k', label = "BDF2")
plt.plot(bdf3['Error'], bdf3['NFE'], 'o', markersize=20, color='tab:brown', label = "BDF3")
plt.plot(bdf4['Error'], bdf4['NFE'], 'o', markersize=20, color='tab:green', label = "BDF4")
plt.plot(cn['Error'], cn['NFE'], 'o', markersize=20, color='tab:blue', label = "Crank Nicolson")
plt.plot(euler['Error'], euler['NFE'], 'o', markersize=20, color='tab:purple', label = "Backward Euler")


plt.xlabel("Error", fontsize=35+10) # Terminal error?
plt.ylabel("NFE", fontsize=35+10)
#plt.legend(loc='lower right', fontsize=35+0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=35+0)
plt.tight_layout()
plt.savefig('./out/NFE_1D_Diffusion_v2.pdf')


"""
Example 3(Convergence): 
    -ode: dh/dt=-L*h(t,x)
    - L: discrete laplacian
"""

optim = pd.read_csv('./out/optimizer_convergence.csv')
sns.set_style()

g=sns.relplot(data=df,x='Iteration',y='err',
            hue='Method',kind='line')
g.set(yscale='log')
plt.savefig('./out/optimizer_convergence.pdf')