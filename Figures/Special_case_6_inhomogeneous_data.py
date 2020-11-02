"""
The goal of this script is to showcase kernel inference for a nontrivial case where
the kernel is to be inferred from point measurements and integral measurements.
An unknown nonzero mean is to be estimated jointly with the covariance and
complicates the procedure. This produces a figure showcasing  the kernel inference
procedure and its uses as detailed in the case example nr 6 which deals with 
applications featuring inhomogeneous data. This example features measurements
which may be considered 1 D versions of tomography measurements.
 More details can be found in the paper:
'Inference of instationary covariance functions for optimal estimation in 
spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Simulations
    4. Kernel inference
    5. Plots and illustrations
    
The simulations are based on a fixed random seed, to generate data deviating 
from the one shown in the paper and different for each run, please comment out
the entry 'np.random.seed(x)' in section 1.


"""



"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Import packages

import numpy as np
import numpy.linalg as lina
import scipy.linalg as spla
import copy as copy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


# ii) Define dimensionality

n=100
n_simu_1=1
n_simu_2=100

n_sample_1=15
n_sample_2=15
n_exp=10
n_mu=3


# iii) Define auxiliary quantities

t=np.linspace(0,1,n)
sample_index_1=np.round(np.linspace(0,n-1,n_sample_1))
t_sample=t[sample_index_1.astype(int)]
np.random.seed(0)

tol=10**(-6)


# iv) Define second measurement operator

sample_index_2=np.round(np.linspace(0,n-1,n_sample_2))
Integration=np.zeros([n_sample_2,n])
for k in range(n_sample_2):
    Integration[k,:sample_index_2[k].astype(int)+1]=1/((sample_index_2[k]+1))



"""
     2. Simulations ----------------------------------------------------------
"""


# i) Define true underlying covariance function

d_true=0.5
def cov_fun_true(t1,t2):
    return 1*np.exp(-(lina.norm(t1-t2)/d_true)**2)


# ii) Set up covariance matrix

K_true=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_true[k,l]=cov_fun_true(t[k],t[l])



# iii) Set up mean vector

beta_true=3*np.random.normal(0,1,[n_mu,1])
def mean_fun_true(t,beta,n_mu):
    G=np.zeros([n,n_mu])
    for k in range(n_mu):
        G[:,k]=t**k
        mu= G@beta
        mu=np.reshape(mu,[n])
    return G, mu

G,mu_true=mean_fun_true(t,beta_true,n_mu)
G_measured_1=G[sample_index_1.astype(int),:]
G_measured_2=Integration@G
G_list=[G_measured_1,G_measured_2]

# iv) Generate simulations

x_simu_1=np.zeros([n,n_simu_1])
x_simu_2=np.zeros([n,n_simu_2])

for k in range(n_simu_1):
    x_simu_1[:,k]=np.random.multivariate_normal(mu_true,K_true)
    
for k in range(n_simu_2):
    x_simu_2[:,k]=np.random.multivariate_normal(mu_true,K_true)

x_measured_1=x_simu_1[sample_index_1.astype(int),:]
x_measured_2=Integration@x_simu_2
x_measured_list=[x_measured_1,x_measured_2]


# v) Create empirical covariance matrix

S_emp_full=(1/n_simu_1)*(x_simu_1@x_simu_1.T)
S_emp_1=(1/n_simu_1)*(x_measured_1@x_measured_1.T)
S_emp_2=(1/n_simu_2)*(x_measured_2@x_measured_2.T)



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

Psi_1=U_p_cut[sample_index_1.astype(int),:]
Psi_2=Integration@U_p_cut
Psi_list=[Psi_1, Psi_2]


# iv) Kernel inference iteration

gamma_0=copy.copy(Lambda_p_cut)

import sys
sys.path.append("..")
import KI
beta, mu_beta, gamma_tilde, C_gamma_tilde, gamma, KI_logfile = KI.\
Kernel_inference_inhomogeneous(x_measured_list,gamma_0,Psi_list,r, G_list=G_list, max_iter=20)


# v) Extract important matrices

mu_estimated=G@beta
K_estimated=U_p_cut@gamma@U_p_cut.T


"""
     4. Plots and illustrations ----------------------------------------------
"""


# i) Figure 1 showing covariances

w,h=plt.figaspect(0.25)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(2, 6)
f1_ax1 = fig1.add_subplot(gs1[0:2, 0:2])
f1_ax1.imshow(K_true)
f1_ax1.set_title('True covariance function')
f1_ax1.axis('off')

f1_ax2 = fig1.add_subplot(gs1[0:2, 4:6])
f1_ax2.imshow(K_estimated)
f1_ax2.set_title('Estimated covariance function')
f1_ax2.axis('off')

f1_ax3 = fig1.add_subplot(gs1[0, 2])
f1_ax3.imshow(spla.block_diag(S_emp_1,S_emp_2))
f1_ax3.set_title('Empirical covariance')
f1_ax3.axis('off')

f1_ax4 = fig1.add_subplot(gs1[0, 3])
f1_ax4.imshow(C_gamma_tilde)
f1_ax4.set_title('Estimated covariance')
f1_ax4.axis('off')

f1_ax5 = fig1.add_subplot(gs1[1, 2])
f1_ax5.imshow(Lambda_p_cut)
f1_ax5.set_title('Prior gamma')
f1_ax5.axis('off')

f1_ax6 = fig1.add_subplot(gs1[1, 3])
f1_ax6.imshow(gamma)
f1_ax6.set_title('gamma')
f1_ax6.axis('off')

# Save the figure
# plt.savefig('Special_case_6a_inhomogeneous_data',dpi=400)


# ii) Figure 2 showing means and examples

n_illu=20
x_illu=np.zeros([n,n_illu])

for k in range(n_illu):
    x_illu[:,k]=np.random.multivariate_normal(mu_true,K_true)


fig2=plt.figure(dpi=400,constrained_layout=True)
gs2=fig2.add_gridspec(1,2)

f2_ax1=fig2.add_subplot(gs2[0,0])
f2_ax1.plot(x_illu[:,:n_illu],linestyle='solid',color='0',label='Example realization')
f2_ax1.set_title('Example realizations')
y_min,y_max=plt.ylim()
plt.ylabel('Function value x(t)')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)


f2_ax2 = fig2.add_subplot(gs2[0,1])
f2_ax2.plot(mu_true,linestyle='dashed',color='0',label='True mean')
f2_ax2.plot(mu_estimated,linestyle='solid',color='0',label='Estimated mean')
f2_ax2.set_title('Mean  comparisons')
plt.ylim(y_min,y_max)
plt.ylabel('Function value mu(t)')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax2.legend(loc='lower right')


# Save the figure
# plt.savefig('Special_case_6b_inhomogeneous_data',dpi=400)
