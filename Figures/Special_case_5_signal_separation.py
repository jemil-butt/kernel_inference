"""
The goal of this script is to showcase kernel inference for the task of signal 
separation by means of splitting data into parts with different correlation
structure. This requires optimally estimating a smooth and a ragged correlation
structure first. The subsequent signal separation produces figure 6 of the paper
'Inference of instationary covariance functions for optimal estimation in 
spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Create covariance matrices
    3. Simulation of autocorrelated data
    4. Kernel inference
    5. Signal separation
    6. Plots and illustrations
    
The simulations are based on a fixed random seed, to generate data deviating 
from the one shown in the paper and different for each run, please comment out
the entry 'np.random.seed(x)' in section 1.

"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import numpy.linalg as lina
import scipy.linalg as spla
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})




# ii) Definition of auxiliary quantities

n=20
n_sample=5
n_simu=150

t=np.linspace(0,1,n)
sample_index=np.round(np.linspace(0,n-1,n_sample))
t_sample=t[sample_index.astype(int)]
np.random.seed(0)

tol=10**(-6)



"""
    2. Create covariance matrices --------------------------------------------
"""


# i) Define auxiliary covariance function

d_sqexp=0.05
def cov_fun_sqexp(t1,t2):
    return (1/n**2)*np.exp(-(lina.norm(t1-t2)/d_sqexp)**2)

K_sqexp=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_sqexp[k,l]=cov_fun_sqexp(t[k],t[l])


# ii) Introduce constrained behavior

Nabla=np.delete(np.eye(n)-np.roll(np.eye(n),1,1),n-1,0)
Delta=np.delete(Nabla.T@Nabla,[0,n-1],0)

L=np.zeros([3,n])
L[0,0]=1; L[2,n-1]=1
L[1,np.round(n/2-1).astype(int)]=1; L[1,np.round(n/2).astype(int)]=-0.9

A_constraints=np.vstack((Delta,L))
K_sqexp_mod=np.delete(K_sqexp,[0,n-1],0)
K_sqexp_mod=np.delete(K_sqexp_mod,[0,n-1],1)
K_constrained=spla.block_diag(K_sqexp_mod,np.zeros([3,3]))


# iii) Solve A_c K_x A_c.T=K_c and compose

K_x_signal=lina.pinv(A_constraints)@K_constrained@lina.pinv(A_constraints).T
K_x_noise=0.1*np.eye(np.shape(K_x_signal)[0])

K_x_measurement=K_x_signal+K_x_noise



"""
    3. Simulation of autocorrelated data -------------------------------------
"""


# i) Initialize

x_signal_simu=np.zeros([n,n_simu])
x_noise_simu=np.zeros([n,n_simu])


# ii) Simulate and compose

for k in range(n_simu):
    x_signal_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_x_signal)
    x_noise_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_x_noise)

x_measured=x_signal_simu+x_noise_simu

S_emp=(1/n_simu)*(x_measured@x_measured.T)

"""
    4. Kernel inference ------------------------------------------------------
"""


# i) Specify model

n_exp=5
d_sqexp=0.2
def cov_fun_sqexp(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_sqexp)**2)

K_sqexp=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_sqexp[k,l]=cov_fun_sqexp(t[k],t[l])

[U_p,Lambda_p,V_p]=lina.svd(K_sqexp)
U_p_cut=U_p[:,:n_exp]
Lambda_p_cut=np.diag(Lambda_p[:n_exp])

Psi=U_p_cut


# ii) Create linear constraints

# Step 1 Augmentation
Psi_tilde=np.hstack((np.eye(n),Psi))
Lambda_tilde=spla.block_diag(np.eye(n),Lambda_p_cut)
n_total=n_exp+n

# Step 2 Diagonal constraints
Diag_mat=spla.block_diag(np.eye(n),np.zeros([n_exp,n_exp]))

Index_tuple_diag=np.nonzero(Diag_mat)
diag_index_1=Index_tuple_diag[0]
diag_index_2=Index_tuple_diag[1]
lin_ind_diag=np.ravel_multi_index(Index_tuple_diag,[n_total,n_total])
n_diag=len(diag_index_1)

A_dg=np.zeros([n_diag-1,n_total**2])

for k in range(n_diag-1):
    A_dg[k,lin_ind_diag[k]]=1
    A_dg[k,lin_ind_diag[k+1]]=-1


# Step 3 Off diagonal constraints
Indicator_mat=spla.block_diag(np.eye(n),np.ones([n_exp,n_exp]))    
Off_diag_mat=np.ones([n_total,n_total])-Indicator_mat

Index_tuple=np.nonzero(Off_diag_mat)
nd_index_1=Index_tuple[0]
nd_index_2=Index_tuple[1]
lin_ind=np.ravel_multi_index(Index_tuple,[n_total,n_total])
n_nondiag=len(nd_index_1)
    
A_nd=np.zeros([n_nondiag,n_total**2])

for k in range(n_nondiag):
    A_nd[k,lin_ind[k]]=1
    
A_c=np.vstack((A_dg,A_nd))


# iii) Perform inference

r=0
import sys
sys.path.append("..")
import KI
beta, mu_beta, gamma_tilde, C_gamma_tilde, KI_logfile = KI.Kernel_inference_homogeneous(x_measured, Lambda_tilde, Psi_tilde,r, A=A_c)


# iv) Assemble solutions
K_gamma=C_gamma_tilde
K_signal_est=Psi@gamma_tilde[np.ix_(range(n,n+n_exp),range(n,n+n_exp))]@Psi.T
K_noise_est=gamma_tilde[np.ix_(range(n),range(n))]



"""
    5. Signal separation  ---------------------------------------------------
"""


# i) Perform signal separation
x_est_noise=K_noise_est@lina.pinv(K_noise_est+K_signal_est,rcond=tol,hermitian=True)@x_measured
x_est_signal=K_signal_est@lina.pinv(K_noise_est+K_signal_est,rcond=tol,hermitian=True)@x_measured



"""
    6. Plots and illustrations -----------------------------------------------
"""



# i) Invoke figure 1

n_illu=10
w,h=plt.figaspect(0.9)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(3, 3)


# ii) First row : Ground truth

f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.imshow(K_x_measurement)
f1_ax1.set_title('True covariance function')
f1_ax1.axis('off')

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.imshow(K_x_signal,vmin=0,vmax=0.3)
f1_ax2.set_title('True signal covariance')
f1_ax2.axis('off')

f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.imshow(K_x_noise,vmin=0,vmax=0.1)
f1_ax3.set_title('True noise covariance')
f1_ax3.axis('off')


# iii) Second row : Data and estimations

f1_ax4 = fig1.add_subplot(gs1[1,0])
f1_ax4.plot(t,x_measured[:,:n_illu],linestyle='solid',color='0.0')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Time t')
f1_ax4.set_title('Example realizations')

f1_ax5 = fig1.add_subplot(gs1[1,1])
f1_ax5.imshow(K_signal_est,vmin=0,vmax=0.3)
f1_ax5.set_title('Estimated signal covariance')
f1_ax5.axis('off')

f1_ax6 = fig1.add_subplot(gs1[1,2])
f1_ax6.imshow(K_noise_est,vmin=0,vmax=0.1)
f1_ax6.set_title('Estimated noise covariance')
f1_ax6.axis('off')


# iv) Third row : Example signal separation

f1_ax7 = fig1.add_subplot(gs1[2,0])
f1_ax7.scatter(t,x_measured[:,0],facecolors='none',edgecolors='0',label='Data points')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
y_min,y_max=plt.ylim()
plt.xlabel('Time t')
plt.ylabel('Function value x(t)')
f1_ax7.set_title('Example realization')


f1_ax8 = fig1.add_subplot(gs1[2,1])
f1_ax8.scatter(t,x_noise_simu[:,0],facecolors='none',edgecolors='0',label='Noise')
f1_ax8.plot(t,x_signal_simu[:,0],linestyle='solid',color='0.0')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.ylim(y_min,y_max)
plt.xlabel('Time t')
f1_ax8.set_title('Ground truth split')


f1_ax9 = fig1.add_subplot(gs1[2,2])
f1_ax9.scatter(t,x_est_noise[:,0],facecolors='none',edgecolors='0',label='Noise')
f1_ax9.plot(t,x_est_signal[:,0],linestyle='solid',color='0.0')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.ylim(y_min,y_max)
plt.xlabel('Time t')
f1_ax9.set_title('Estimated split')


# Save the figure
plt.savefig('Figure_6',dpi=400)






















