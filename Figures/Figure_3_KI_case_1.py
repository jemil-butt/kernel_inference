"""
The goal of this script is to showcase kernel inference for a simple case where
the kernel is to be inferred from point measurements and neither linear constraints
nor nonzero mean or inhomogeneities complicate the procedure. This produces
figures 3 of the paper
'Inference of instationary covariance functions for optimal estimation in 
spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Create covariance matrices
    3. Simulation of autocorrelated data
    4. Kernel inference
    5. Optimal estimation
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
plt.rcParams.update({'font.size': 12})




# ii) Definition of auxiliary quantities

n=100
n_sample=10
n_simu=100

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
L[0,0]=1; L[2,n-1]=1; L[1,np.round(n/2-1).astype(int)]=1

A_constraints=np.vstack((Delta,L))
K_sqexp_mod=np.delete(K_sqexp,[0,n-1],0)
K_sqexp_mod=np.delete(K_sqexp_mod,[0,n-1],1)
K_constrained=spla.block_diag(K_sqexp_mod,np.zeros([3,3]))


# iii) Solve A_c K_x A_c.T=K_c

K_x=lina.pinv(A_constraints)@K_constrained@lina.pinv(A_constraints).T



"""
    3. Simulation of autocorrelated data -------------------------------------
"""


# i) Draw from a distribution with covariance matrix K_x

x_simu=np.zeros([n,n_simu])
for k in range(n_simu):
    x_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_x)

x_measured=x_simu[sample_index.astype(int),:]

S_emp=(1/n_simu)*(x_simu@x_simu.T)
S_emp_measured=(1/n_simu)*(x_measured@x_measured.T)




"""
    4. Kernel inference ------------------------------------------------------
"""


# i) Preparation

r=1
n_exp=5

[U_p,Lambda_p,V_p]=lina.svd(K_x,hermitian=True)
U_p_cut=U_p[:,:n_exp]
Psi=U_p_cut[sample_index.astype(int),:]
Lambda_p_cut=np.diag(Lambda_p[:n_exp])

# ii) Execute inference

import sys
sys.path.append("..")
import KI
beta, mu, gamma, C_gamma, KI_logfile = KI.Kernel_inference_homogeneous(x_measured,Lambda_p_cut,Psi,r)



"""
    5. Optimal estimation  ---------------------------------------------------
"""


# i) Auxiliary quantities

n_datapoints= 4
datapoint_index=np.sort(np.random.choice(range(n),size=n_datapoints))
t_datapoints=t[datapoint_index.astype(int)]
x_datapoints=x_simu[datapoint_index.astype(int),:]


# ii) Interpolate using squared exponential

d_sqexp_interpolate=0.2
def cov_fun_sqexp_interpolate(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_sqexp_interpolate)**2)

K_sqexp_interpolate_sample=np.zeros([n_datapoints,n_datapoints])
K_sqexp_interpolate_subset=np.zeros([n,n_datapoints])
for k in range(n_datapoints):
    for l in range(n_datapoints):
        K_sqexp_interpolate_sample[k,l]=cov_fun_sqexp_interpolate(t_datapoints[k],t_datapoints[l])
        
for k in range(n):
    for l in range(n_datapoints):
        K_sqexp_interpolate_subset[k,l]=cov_fun_sqexp_interpolate(t[k],t_datapoints[l])
        
x_est_K_sqexp=K_sqexp_interpolate_subset@lina.pinv(K_sqexp_interpolate_sample,rcond=tol,hermitian=True)@x_datapoints


# iii) Interpolate using inferred kernel

K_gamma=U_p_cut@gamma@U_p_cut.T
K_gamma_sample=K_gamma[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_gamma_subset=K_gamma[:,datapoint_index.astype(int)]

x_est_K_gamma=K_gamma_subset@lina.pinv(K_gamma_sample,rcond=tol,hermitian=True)@x_datapoints


# iv) Interpolate using true kernel

K_true_sample=K_x[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_true_subset=K_x[:,datapoint_index.astype(int)]

x_est_K_true=K_true_subset@lina.pinv(K_true_sample,rcond=tol,hermitian=True)@x_datapoints




"""
    6. Plots and illustrations -----------------------------------------------
"""


# i) Auxiliary definitions

zero_line=np.zeros([n,1])


# ii) Invoke figure 1

n_plot=15
w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Underlying covariance function
f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.imshow(K_x)
plt.ylabel('Time t')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Covariance function')


# Location 1,2 Example realizations
f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(t,x_simu[:,1:n_plot],linestyle='solid',color='0',label='Estimate sqexp')

y_min,y_max=plt.ylim()
plt.vlines(t_sample,y_min,y_max,color='0.75',linestyle='dashed')
plt.ylabel('Function value x(t)')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Example realizations')


# Location 1,3 Plot of the empirical covariance matrix
f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.imshow((1/n_sample)*x_measured@x_measured.T)
plt.ylabel('Time t')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax3.set_title('Empirical covariance')


# Save the figure
# plt.savefig('Figure_3a',dpi=400)




# iii) Invoke figure 2

n_plot=15
n_illu=5
w,h=plt.figaspect(0.35)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig2.add_gridspec(4, 6)


f2_ax1 = fig2.add_subplot(gs2[0:2, 0:2])
f2_ax1.imshow(K_x)
f2_ax1.set_title('True covariance function')
f2_ax1.axis('off')

f2_ax2 = fig2.add_subplot(gs2[0:2, 4:6])
f2_ax2.imshow(K_gamma)
f2_ax2.set_title('Estimated covariance function')
f2_ax2.axis('off')

f2_ax3 = fig2.add_subplot(gs2[0, 2])
f2_ax3.imshow(S_emp_measured)
f2_ax3.set_title('Empirical covariance')
f2_ax3.axis('off')

f2_ax4 = fig2.add_subplot(gs2[0, 3])
f2_ax4.imshow(C_gamma)
f2_ax4.set_title('Estimated covariance')
f2_ax4.axis('off')

f2_ax5 = fig2.add_subplot(gs2[1, 2])
f2_ax5.imshow(Lambda_p_cut)
f2_ax5.set_title('Prior gamma')
f2_ax5.axis('off')

f2_ax6 = fig2.add_subplot(gs2[1, 3])
f2_ax6.imshow(gamma)
f2_ax6.set_title('Inferred gamma')
f2_ax6.axis('off')


# Save the figure
# plt.savefig('Figure_3b',dpi=400)





# iii) Invoke figure 3

w,h=plt.figaspect(0.25)
fig3 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs3 = fig3.add_gridspec(1, 3)

# Location 1.2 Estimations using squared exponential covariance
f3_ax1 = fig3.add_subplot(gs3[0,1])
f3_ax1.scatter(t_datapoints,x_datapoints[:,0],facecolors='none',edgecolors='0',label='Data points')
for k in range(n_illu-1):
    f3_ax1.scatter(t_datapoints,x_datapoints[:,k+1],facecolors='none',edgecolors='0')
    
sqexp_est = f3_ax1.plot(t,x_est_K_sqexp[:,:n_illu],linestyle='solid',color='0',label='Estimate sqexp cov')
plt.setp(sqexp_est[1:], label="_")
true_est = f3_ax1.plot(t,x_est_K_true[:,:n_illu],linestyle='dotted',color='0.65',label='Estimate true cov')
plt.setp(true_est[1:], label="_")
f3_ax1.plot(t,zero_line,linestyle='dotted',color='0.5')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Time t')
f3_ax1.set_title('Estimations using squared exp. covariance')
f3_ax1.legend(loc='lower right')



# Location 1.3 Estimations using inferred covariance
f3_ax2 = fig3.add_subplot(gs3[0,2])
f3_ax2.scatter(t_datapoints,x_datapoints[:,0],facecolors='none',edgecolors='0',label='Data points')
for k in range(n_illu-1):
    f3_ax2.scatter(t_datapoints,x_datapoints[:,k+1],facecolors='none',edgecolors='0')
    
gamma_est = f3_ax2.plot(t,x_est_K_gamma[:,:n_illu],linestyle='solid',color='0',label='Estimate inferred cov')
plt.setp(gamma_est[1:], label="_")
true_est = f3_ax2.plot(t,x_est_K_true[:,:n_illu],linestyle='dotted',color='0.65',label='Estimate true cov')
plt.setp(true_est[1:], label="_")
f3_ax2.plot(t,zero_line,linestyle='dotted',color='0.5')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Time t')
f3_ax2.set_title('Estimations using inferred covariance')
f3_ax2.legend(loc='lower right')



# Location 1.1 Example realizations
f3_ax3 = fig3.add_subplot(gs3[0,0])
f3_ax3.plot(t,x_simu[:,1:n_plot],linestyle='solid',color='0',label='Estimate sqexp')

y_min,y_max=plt.ylim()
plt.vlines(t_sample,y_min,y_max,color='0.75',linestyle='dashed')
plt.ylabel('Function value x(t)')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax3.set_title('Example realizations')


# Save the figure
# plt.savefig('Figure_3c',dpi=400)




























