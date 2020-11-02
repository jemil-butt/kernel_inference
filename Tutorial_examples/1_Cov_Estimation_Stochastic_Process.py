"""
The goal of this script is to illustrate the results of applying kernel inference
in its simplest form to derive a best guess for a covariance function.
For this, do the following:
    1. Definitions and imports
    2. Simulations
    3. Perform kernel inference
    4. Plots and illustrations
"""



"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Import packages

import numpy as np
import numpy.linalg as lina
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})


# ii) Define auxiliary quantities

n=100
n_simu=80
n_sample=30
n_exp=10

t=np.linspace(0,1,n)
sample_index=np.round(np.linspace(0,n-1,n_sample))
t_sample=t[sample_index.astype(int)]

tol=10**(-10)



"""
     2. Simulations ----------------------------------------------------------
"""


# i) Define true underlying covariance function

d_true=0.5
def cov_fun_true(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_true)**2)


# ii) Set up Covariance matrix

K_true=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_true[k,l]=cov_fun_true(t[k],t[l])

mu=np.zeros(n)


# iii) Generate simulations

x_simu=np.zeros([n,n_simu])
for k in range(n_simu):
    x_simu[:,k]=np.random.multivariate_normal(mu,K_true)

x_measured=x_simu[sample_index.astype(int),:]


# iv) Create empirical covariance matrix

S_emp_full=(1/n_simu)*(x_simu@x_simu.T)
S_emp=(1/n_simu)*(x_measured@x_measured.T)



"""
     3. Perform kernel inference ---------------------------------------------
"""


# i) Create prior

r=2

d_prior=0.2
def cov_fun_prior(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_prior)**2)

K_prior=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_prior[k,l]=cov_fun_prior(t[k],t[l])


# ii) Set up matrices

[U_p,Lambda_p,V_p]=lina.svd(K_prior)
Lambda_p=np.diag(Lambda_p)

U_p_cut=U_p[:,0:n_exp]
Lambda_p_cut=Lambda_p[0:n_exp,0:n_exp]
L_pinv=lina.pinv(Lambda_p_cut,rcond=tol)

Psi=U_p_cut[sample_index.astype(int),:]

S_psi=lina.pinv(Psi,rcond=tol)@S_emp@lina.pinv(Psi.T,rcond=tol)


# iii) Kernel inference iteration

gamma_0=Lambda_p_cut
gamma=gamma_0

import sys
sys.path.append("..")
import KI
beta, mu_beta, gamma, C_gamma, KI_logfile=KI.Kernel_inference_homogeneous(x_measured,gamma_0,Psi,r)


# v) Extract important matrices

C_gamma=Psi@gamma@Psi.T
K_estimated=U_p_cut@gamma@U_p_cut.T



"""
     4. Plots and illustrations ----------------------------------------------
"""


fig1 = plt.figure(dpi=200,constrained_layout=True)
gs = fig1.add_gridspec(2, 6)
f1_ax1 = fig1.add_subplot(gs[0:2, 0:2])
f1_ax1.imshow(K_true)
f1_ax1.set_title('True covariance function')
f1_ax1.axis('off')

f1_ax2 = fig1.add_subplot(gs[0:2, 4:6])
f1_ax2.imshow(K_estimated)
f1_ax2.set_title('Estimated covariance function')
f1_ax2.axis('off')

f1_ax3 = fig1.add_subplot(gs[0, 2])
f1_ax3.imshow(S_emp)
f1_ax3.set_title('Empirical covariance')
f1_ax3.axis('off')

f1_ax4 = fig1.add_subplot(gs[0, 3])
f1_ax4.imshow(C_gamma)
f1_ax4.set_title('Estimated covariance')
f1_ax4.axis('off')

f1_ax5 = fig1.add_subplot(gs[1, 2])
f1_ax5.imshow(Lambda_p_cut)
f1_ax5.set_title('Prior gamma')
f1_ax5.axis('off')

f1_ax6 = fig1.add_subplot(gs[1, 3])
f1_ax6.imshow(gamma)
f1_ax6.set_title('gamma')
f1_ax6.axis('off')

















