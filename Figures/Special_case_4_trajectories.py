"""
The goal of this script is to showcase kernel inference for a simple trajectory
estimation task for which we assume stochastic independence of the two processes
generating x and y coordinates respectively. This produces a figure showcasing 
the kernel inference procedure and its uses as detailed in the case example nr
5 which deals with multivariate applications. More details can be found in the
paper:
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
n_sample=20
n_simu=100

t=np.linspace(0,1,n)
sample_index=np.round(np.linspace(0,n-1,n_sample))
t_sample=t[sample_index.astype(int)]
np.random.seed(0)

tol=10**(-4)



"""
    2. Create covariance matrices --------------------------------------------
"""


# i) Define covariance functions

d_x=0.1
d_y=0.05

def cov_fun_x(t1,t2):
    return 1*np.exp(-(lina.norm(t1-t2)/d_x)**2)

def cov_fun_y(t1,t2):
    return 1.5*np.exp(-(lina.norm(t1-t2)/d_y)**2)


# ii) Create covariance matrices

K_x=np.zeros([n,n])
K_y=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_x[k,l]=cov_fun_x(t[k],t[l])
        K_y[k,l]=cov_fun_y(t[k],t[l])
        
        
# iii) Introduce constrained behavior

# weight_mat=np.ones([n,n])
# for k in range(n):
#     for l in range(n):
#         if np.min([k,l])/(np.round(n/3))<=1:
#             weight_mat[k,l]=np.min([k,l])/(np.round(n/3))

# K_x=weight_mat*K_x
# K_y=weight_mat*K_y

Nabla=np.delete(np.eye(n)-np.roll(np.eye(n),1,1),n-1,0)

L=np.zeros([1,n])
L[0,0]=1; 

A_constraints=np.vstack((Nabla,L))
K_x_mod=np.delete(K_x,[n-1],0)
K_x_mod=np.delete(K_x_mod,[n-1],1)
K_x_constrained=spla.block_diag(K_x_mod,np.zeros([1,1]))

K_y_mod=np.delete(K_y,[n-1],0)
K_y_mod=np.delete(K_y_mod,[n-1],1)
K_y_constrained=spla.block_diag(K_y_mod,np.zeros([1,1]))


# iv) Solve A_c K_x A_c.T=K_c

K_x=lina.pinv(A_constraints)@K_x_constrained@lina.pinv(A_constraints).T
K_y=lina.pinv(A_constraints)@K_y_constrained@lina.pinv(A_constraints).T



"""
    3. Simulation of autocorrelated data -------------------------------------
"""


# i) Draw from a distribution with covariance matrix K_x

x_simu=np.zeros([n,n_simu])
y_simu=np.zeros([n,n_simu])
for k in range(n_simu):
    x_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_x)
    y_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_y)

x_measured=x_simu[sample_index.astype(int),:]
y_measured=y_simu[sample_index.astype(int),:]

S_emp_x=(1/n_simu)*(x_simu@x_simu.T)
S_emp_measured_x=(1/n_simu)*(x_measured@x_measured.T)

S_emp_y=(1/n_simu)*(y_simu@y_simu.T)
S_emp_measured_y=(1/n_simu)*(y_measured@y_measured.T)



"""
    4. Kernel inference ------------------------------------------------------
"""


# i) Preparation

r=2
n_exp=10

d_sqexp=0.3
def cov_fun_exp(t1,t2):
    return (1/n**2)*np.exp(-(lina.norm(t1-t2)/d_sqexp)**1)

K_exp=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_exp[k,l]=cov_fun_exp(t[k],t[l])

[U_p,Lambda_p,V_p]=lina.svd(K_exp,hermitian=True)
U_p_cut=U_p[:,:n_exp]
Psi=U_p_cut[sample_index.astype(int),:]
Lambda_p_cut=np.diag(Lambda_p[:n_exp])


# ii) Execute inference

import sys
sys.path.append("..")
import KI
beta_x, mu_x, gamma_x, C_gamma_x, KI_logfile_x = KI.Kernel_inference_homogeneous(x_measured,Lambda_p_cut,Psi,r,max_iter=300)
beta_y, mu_y, gamma_y, C_gamma_y, KI_logfile_y = KI.Kernel_inference_homogeneous(y_measured,Lambda_p_cut,Psi,r, max_iter=300)



"""
    5. Optimal estimation  ---------------------------------------------------
"""


# i) Auxiliary quantities

n_datapoints= 10
datapoint_index=np.sort(np.random.choice(range(n),size=n_datapoints))
t_datapoints=t[datapoint_index.astype(int)]
x_datapoints=x_simu[datapoint_index.astype(int),:]
y_datapoints=y_simu[datapoint_index.astype(int),:]


# ii) Interpolate x using inferred kernel

K_gamma_x=U_p_cut@gamma_x@U_p_cut.T
K_gamma_x_sample=K_gamma_x[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_gamma_x_subset=K_gamma_x[:,datapoint_index.astype(int)]

x_est_K_gamma_x=K_gamma_x_subset@lina.pinv(K_gamma_x_sample,rcond=tol,hermitian=True)@x_datapoints


# iii) Interpolate y using inferred kernel

K_gamma_y=U_p_cut@gamma_y@U_p_cut.T
K_gamma_y_sample=K_gamma_y[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_gamma_y_subset=K_gamma_y[:,datapoint_index.astype(int)]

y_est_K_gamma_y=K_gamma_y_subset@lina.pinv(K_gamma_y_sample,rcond=tol,hermitian=True)@y_datapoints



# iv) Interpolate using true kernel

K_x_true_sample=K_x[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_x_true_subset=K_x[:,datapoint_index.astype(int)]
x_est_K_x_true=K_x_true_subset@lina.pinv(K_x_true_sample,rcond=tol,hermitian=True)@x_datapoints

K_y_true_sample=K_y[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_y_true_subset=K_y[:,datapoint_index.astype(int)]
y_est_K_y_true=K_y_true_subset@lina.pinv(K_y_true_sample,rcond=tol,hermitian=True)@y_datapoints


# v) Interpolate using generic squared exponential

K_exp_sample=K_exp[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_exp_subset=K_exp[:,datapoint_index.astype(int)]
x_est_K_exp=K_exp_subset@lina.pinv(K_exp_sample,rcond=tol,hermitian=True)@x_datapoints
y_est_K_exp=K_exp_subset@lina.pinv(K_exp_sample,rcond=tol,hermitian=True)@y_datapoints




"""
    6. Plots and illustrations -----------------------------------------------
"""


# i) Auxiliary definitions

zero_line=np.zeros([n,1])
K=spla.block_diag(K_x,K_y)
K_gamma=spla.block_diag(K_gamma_x,K_gamma_y)
S_emp=spla.block_diag(S_emp_x,S_emp_y)

K_gamma_x_sample=K_gamma_x[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_gamma_y_sample=K_gamma_y[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_gamma_sample=spla.block_diag(K_gamma_x_sample,K_gamma_y_sample
                               )
S_emp_measured=spla.block_diag(S_emp_measured_x,S_emp_measured_y)
gamma=spla.block_diag(gamma_x,gamma_y)


# ii) Invoke figure 1

n_plot=2
w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Underlying covariance function
f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.imshow(K)
plt.ylabel('Locations x,y')
plt.xlabel('Locations x,y')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Covariance function')


# Location 1,2 Example realizations
f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(x_simu[:,1:n_plot],y_simu[:,1:n_plot],linestyle='solid',color='0')

y_min,y_max=plt.ylim()

plt.ylabel('Function value y(t)')
plt.xlabel('Function value x(t)')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Example realizations')


# Location 1,3 Plot of the empirical covariance matrix
f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.imshow(S_emp)
plt.ylabel('Locations x,y')
plt.xlabel('Locations x,y')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax3.set_title('Empirical covariance')


# Save the figure
# plt.savefig('Special_case_4a_trajectories',dpi=400)




# iii) Invoke figure 2

n_plot=2
n_illu=1
w,h=plt.figaspect(0.35)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig2.add_gridspec(4, 6)


f2_ax1 = fig2.add_subplot(gs2[0:2, 0:2])
f2_ax1.imshow(K)
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
f2_ax4.imshow(K_gamma_sample)
f2_ax4.set_title('Estimated covariance')
f2_ax4.axis('off')

f2_ax5 = fig2.add_subplot(gs2[1, 2])
f2_ax5.imshow(spla.block_diag(Lambda_p_cut,Lambda_p_cut))
f2_ax5.set_title('Prior gamma')
f2_ax5.axis('off')

f2_ax6 = fig2.add_subplot(gs2[1, 3])
f2_ax6.imshow(gamma)
f2_ax6.set_title('Inferred gamma')
f2_ax6.axis('off')


# Save the figure
# plt.savefig('Special_case_4b_trajectories',dpi=400)




# iii) Invoke figure 3

w,h=plt.figaspect(0.25)
fig3 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs3 = fig3.add_gridspec(1, 3)

# Location 1.2 Estimations using squared exponential covariance
f3_ax1 = fig3.add_subplot(gs3[0,1])
f3_ax1.scatter(x_datapoints[:,0],y_datapoints[:,0],facecolors='none',edgecolors='0',label='Data points')
for k in range(n_illu-1):
    f3_ax1.scatter(x_datapoints[:,k+1],y_datapoints[:,k+1],facecolors='none',edgecolors='0')
    
exp_est = f3_ax1.plot(x_est_K_exp[:,:n_illu] ,y_est_K_exp[:,:n_illu],linestyle='solid',color='0',label='Estimate sqexp cov')
plt.setp(exp_est[1:], label="_")
true_est = f3_ax1.plot(x_est_K_x_true[:,:n_illu],y_est_K_y_true[:,:n_illu],linestyle='dotted',color='0.65',label='Estimate true cov')
plt.setp(true_est[1:], label="_")
f3_ax1.plot(t,zero_line,linestyle='dotted',color='0.5')
f3_ax1.plot(zero_line,t,linestyle='dotted',color='0.5')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Locations x,y')
f3_ax1.set_title('Estimations using exp. covariance')
f3_ax1.legend(loc='lower right')



# Location 1.3 Estimations using inferred covariance

f3_ax2 = fig3.add_subplot(gs3[0,2])
f3_ax2.scatter(x_datapoints[:,0],y_datapoints[:,0],facecolors='none',edgecolors='0',label='Data points')
for k in range(n_illu-1):
    f3_ax2.scatter(x_datapoints[:,k+1],y_datapoints[:,k+1],facecolors='none',edgecolors='0')
    
KI_est = f3_ax2.plot(x_est_K_gamma_x[:,:n_illu] ,y_est_K_gamma_y[:,:n_illu],linestyle='solid',color='0',label='Estimate KI cov')
plt.setp(KI_est[1:], label="_")
true_est = f3_ax2.plot(x_est_K_x_true[:,:n_illu],y_est_K_y_true[:,:n_illu],linestyle='dotted',color='0.65',label='Estimate true cov')
plt.setp(true_est[1:], label="_")
f3_ax2.plot(t,zero_line,linestyle='dotted',color='0.5')
f3_ax2.plot(zero_line,t,linestyle='dotted',color='0.5')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Locations x,y')
f3_ax2.set_title('Estimations using inferred covariance')
f3_ax2.legend(loc='lower right')



# Location 1.1 Example realizations
f3_ax3 = fig3.add_subplot(gs3[0,0])
f3_ax3.plot(x_simu[:,1:n_plot],y_simu[:,1:n_plot],linestyle='solid',color='0',label='Estimate sqexp')

y_min,y_max=plt.ylim()
plt.ylabel('Locations x,y')
plt.xlabel('Locations x,y')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax3.set_title('Example realizations')


# Save the figure
# plt.savefig('Special_case_4c_trajectories',dpi=400)




























