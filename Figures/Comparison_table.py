"""
The goal of this script is to showcase the performance of kernel inference in
comparison to other, more traditional methods for the purpose of estimation and
interpolation. The subsequent root mean square errors produces can be assembled
to produce a table appearing in the paper 'Inference of instationary covariance 
functions for optimal estimation in spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Create covariance matrices
    3. Simulation for model training
    4. Infer best model parameters: Parametric
    5. Infer best model parameters: Bochner
    6. Kernel inference
    7. Simulation and RMSE
    8. Plots and illustrations
    
The simulations are based on a fixed random seed, to generate data deviating 
from the one shown in the paper and different for each run, please comment out
the entry 'np.random.seed(x)' in section 1.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import sys
sys.path.append("..")

import numpy as np
import numpy.linalg as lina
import matplotlib.pyplot as plt
import scipy as sc
import scipy.optimize as scopt
plt.rcParams.update({'font.size': 6})

K_nondiag_mercer_1=np.load("../Data/K_nondiag_Mercer_1.npy")
K_nondiag_mercer_2=np.load("../Data/K_nondiag_Mercer_1.npy")


# ii) Auxiliary quantities

n=300
n_sample=5
n_simu=1
n_test=1000

t=np.linspace(0,1,n)
sample_index=np.round(np.linspace(n/4,3*n/4,n_sample))
t_sample=t[sample_index.astype(int)]
np.random.seed(0)

tol=10**(-6)


"""
    2. Create covariance matrices
"""


# i) Define parameters

d_sqexp=0.2
d_exp=0.5
n_exp_Bochner=10


# ii) Create covariance functions

def cov_fun_sqexp_true(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_sqexp)**2)

def cov_fun_exp_true(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_exp))

def cov_fun_bb_true(t1,t2):
    return np.min((t1,t2))-t1*t2


# iii) Assemble matrices

K_sqexp=np.zeros([n,n])
K_exp=np.zeros([n,n])
K_bb=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp[k,l]=cov_fun_sqexp_true(t[k],t[l])
        K_exp[k,l]=cov_fun_exp_true(t[k],t[l])
        K_bb[k,l]=cov_fun_bb_true(t[k],t[l])
        
        



"""
    3. Simulation for model training -----------------------------------------
"""


# i) Initialization

x_simu_sqexp=np.zeros([n,n_simu])
x_simu_exp=np.zeros([n,n_simu])
x_simu_bb=np.zeros([n,n_simu])
x_simu_ndm_1=np.zeros([n,n_simu])
x_simu_ndm_2=np.zeros([n,n_simu])


# ii) Simulate 

for k in range(n_simu):
    x_simu_sqexp[:,k]=np.random.multivariate_normal(np.zeros([n]),K_sqexp)
    x_simu_exp[:,k]=np.random.multivariate_normal(np.zeros([n]),K_exp)
    x_simu_bb[:,k]=np.random.multivariate_normal(np.zeros([n]),K_bb)
    x_simu_ndm_1[:,k]=np.random.multivariate_normal(np.zeros([n]),K_nondiag_mercer_1)
    x_simu_ndm_2[:,k]=np.random.multivariate_normal(np.zeros([n]),K_nondiag_mercer_2)
    
# iii) Empirical covariances

K_emp_sqexp=(1/n_simu)*x_simu_sqexp@x_simu_sqexp.T
K_emp_exp=(1/n_simu)*x_simu_exp@x_simu_exp.T
K_emp_bb=(1/n_simu)*x_simu_bb@x_simu_bb.T
K_emp_ndm_1=(1/n_simu)*x_simu_ndm_1@x_simu_ndm_1.T
K_emp_ndm_2=(1/n_simu)*x_simu_ndm_2@x_simu_ndm_2.T



"""
    4. Infer best model parameters: Parametric -------------------------------
"""


# i) Create empirical correlogram

Dist_mat=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        Dist_mat[k,l]=np.abs(t[k]-t[l])

t_diff=1*t
n_t_diff=np.zeros([n,1])

correlogram_sqexp=np.zeros([n,1])
correlogram_exp=np.zeros([n,1])
correlogram_bb=np.zeros([n,1])
correlogram_ndm_1=np.zeros([n,1])
correlogram_ndm_2=np.zeros([n,1])

for k in range(n):
    Ind_mat=np.isclose(Dist_mat,t[k])
    n_t_diff[k]=np.sum(Ind_mat)

    correlogram_sqexp[k]=np.sum(K_emp_sqexp[Ind_mat])/n_t_diff[k]
    correlogram_exp[k]=np.sum(K_emp_exp[Ind_mat])/n_t_diff[k]
    correlogram_bb[k]=np.sum(K_emp_bb[Ind_mat])/n_t_diff[k]
    correlogram_ndm_1[k]=np.sum(K_emp_ndm_1[Ind_mat])/n_t_diff[k]
    correlogram_ndm_2[k]=np.sum(K_emp_ndm_2[Ind_mat])/n_t_diff[k]
    
    
# ii) Define model covariances

def cov_fun_sqexp_model(t1,t2,sigma,d):
    return sigma*np.exp(-(lina.norm(t1-t2)/d)**2)


def cov_fun_exp_model(t1,t2,sigma,d):
    return sigma*np.exp(-(lina.norm(t1-t2)/d))


# iii) Define objective functions

# First: sqexp dataset
def sqexp_sqexp_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_sqexp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_sqexp)
    return RMSE


def exp_sqexp_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_exp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_sqexp)
    return RMSE

# Second: exp dataset

def sqexp_exp_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_sqexp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_exp)
    return RMSE


def exp_exp_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_exp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_exp)
    return RMSE



# Third: bb dataset

def sqexp_bb_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_sqexp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_bb)
    return RMSE


def exp_bb_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_exp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_bb)
    return RMSE


# Fourth: ndm_1 dataset

def sqexp_ndm_1_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_sqexp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_ndm_1)
    return RMSE


def exp_ndm_1_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_exp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_ndm_1)
    return RMSE


# Fifth: dnm_2 dataset

def sqexp_ndm_2_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_sqexp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_ndm_2)
    return RMSE


def exp_ndm_2_objective(x):
    cov_predicted=np.zeros([n,1])
    for k in range(n):
        cov_predicted[k]=cov_fun_exp_model(t[0],t[k],x[0],x[1])
    
    RMSE=(1/n)*lina.norm(cov_predicted-correlogram_ndm_2)
    return RMSE




# iv) Optimize parameters

# First sqexp dataset
params_optimal_sqexp_sqexp=(scopt.minimize(sqexp_sqexp_objective,[0,1])).x
params_optimal_exp_sqexp=(scopt.minimize(exp_sqexp_objective,[0,1])).x

# Second exp dataset
params_optimal_sqexp_exp=(scopt.minimize(sqexp_exp_objective,[0,1])).x
params_optimal_exp_exp=(scopt.minimize(exp_exp_objective,[0,1])).x

# Third bb dataset
params_optimal_sqexp_bb=(scopt.minimize(sqexp_bb_objective,[0,1])).x
params_optimal_exp_bb=(scopt.minimize(exp_bb_objective,[0,1])).x

# Fourth ndm_1 dataset
params_optimal_sqexp_ndm_1=(scopt.minimize(sqexp_ndm_1_objective,[0,1])).x
params_optimal_exp_ndm_1=(scopt.minimize(exp_ndm_1_objective,[0,1])).x

# Fifth dnm_2 dataset
params_optimal_sqexp_ndm_2=(scopt.minimize(sqexp_ndm_2_objective,[0,1])).x
params_optimal_exp_ndm_2=(scopt.minimize(exp_ndm_2_objective,[0,1])).x






"""
    5. Infer best model parameters: Bochner ----------------------------------
"""


# i) Calculate the basis functions

n_exp_Bochner=30
omega=np.logspace(-1,1,n_exp_Bochner)

def complex_exp(t1,t2,omega):
    return np.exp(2*np.pi*(1j)*omega*(t1-t2))

basis_vectors=np.zeros([n,n_exp_Bochner])+0j*np.zeros([n,n_exp_Bochner])
for k in range(n):
    for l in range(n_exp_Bochner):
        basis_vectors[k,l]=(complex_exp(t[k],0,omega[l]))
        

# ii) Parameter estimation

Bochner_Psi_mat=np.zeros([n,n_exp_Bochner])
for k in range(n):
    for l in range(n_exp_Bochner):
        Bochner_Psi_mat[k,l]=complex_exp(0,t_diff[k],omega[l]) 
                                         
weight_fun_sqexp=(scopt.nnls(Bochner_Psi_mat,np.reshape(correlogram_sqexp,[n])))[0]
weight_fun_exp=(scopt.nnls(Bochner_Psi_mat,np.reshape(correlogram_exp,[n])))[0]
weight_fun_bb=(scopt.nnls(Bochner_Psi_mat,np.reshape(correlogram_bb,[n])))[0]
weight_fun_ndm_1=(scopt.nnls(Bochner_Psi_mat,np.reshape(correlogram_ndm_1,[n])))[0]
weight_fun_ndm_2=(scopt.nnls(Bochner_Psi_mat,np.reshape(correlogram_ndm_2,[n])))[0]


# iii) Assemble to covariance function

K_Bochner_sqexp=np.real(basis_vectors@np.diagflat(weight_fun_sqexp)@basis_vectors.conj().T)
K_Bochner_exp=np.real(basis_vectors@np.diagflat(weight_fun_exp)@basis_vectors.conj().T)
K_Bochner_bb=np.real(basis_vectors@np.diagflat(weight_fun_bb)@basis_vectors.conj().T)
K_Bochner_ndm_1=np.real(basis_vectors@np.diagflat(weight_fun_ndm_1)@basis_vectors.conj().T)
K_Bochner_ndm_2=np.real(basis_vectors@np.diagflat(weight_fun_ndm_2)@basis_vectors.conj().T)



"""
    6. Kernel inference ------------------------------------------------------
"""


# i) Prepare auxiliary quantities

r=2
n_exp=10

d_prior=0.4
def cov_fun_prior(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_prior)**2)

K_prior=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_prior[k,l]=cov_fun_prior(t[k],t[l])
        
[U_p,Lambda_p,V_p]=lina.svd(K_prior)
U_p_cut=U_p[:,:n_exp]
Lambda_p_cut=np.diag(Lambda_p[:n_exp])

Psi=U_p_cut        
        
        
# ii) Perform kernel inference

import KI

beta, mu, gamma_sqexp, C_gamma_sqexp, KI_logfile_sqexp=KI.Kernel_inference_homogeneous(x_simu_sqexp,Lambda_p_cut,Psi,r)
beta, mu, gamma_exp, C_gamma_exp, KI_logfile_exp=KI.Kernel_inference_homogeneous(x_simu_exp,Lambda_p_cut,Psi,r)
beta, mu, gamma_bb, C_gamma_bb, KI_logfile_bb=KI.Kernel_inference_homogeneous(x_simu_bb,Lambda_p_cut,Psi,r)
beta, mu, gamma_ndm_1, C_gamma_ndm_1, KI_logfile_ndm_1=KI.Kernel_inference_homogeneous(x_simu_ndm_1,Lambda_p_cut,Psi,r)
beta, mu, gamma_ndm_2, C_gamma_ndm_2, KI_logfile_ndm_2=KI.Kernel_inference_homogeneous(x_simu_ndm_2,Lambda_p_cut,Psi,r)



"""
    7. Simulation and RMSE ---------------------------------------------------
"""


# i) Set up Matrices for interpolation



# First: sqexp Dataset

K_sqexp_sqexp=np.zeros([n,n])
K_exp_sqexp=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_sqexp[k,l]=cov_fun_sqexp_model(t[k],t[l],params_optimal_sqexp_sqexp[0],params_optimal_sqexp_sqexp[1])
        K_exp_sqexp[k,l]=cov_fun_exp_model(t[k],t[l],params_optimal_exp_sqexp[0],params_optimal_exp_sqexp[1])
        
K_KI_sqexp=C_gamma_sqexp

K_t_sqexp_sqexp=K_sqexp_sqexp[:,sample_index.astype(int)]
K_t_exp_sqexp=K_exp_sqexp[:,sample_index.astype(int)]
K_t_Bochner_sqexp=K_Bochner_sqexp[:,sample_index.astype(int)]
K_t_KI_sqexp=K_KI_sqexp[:,sample_index.astype(int)]

K_ij_sqexp_sqexp=K_sqexp_sqexp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_exp_sqexp=K_exp_sqexp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_Bochner_sqexp=K_Bochner_sqexp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_KI_sqexp=K_KI_sqexp[np.ix_(sample_index.astype(int),sample_index.astype(int))]


# Second: exp Dataset

K_sqexp_exp=np.zeros([n,n])
K_exp_exp=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_exp[k,l]=cov_fun_sqexp_model(t[k],t[l],params_optimal_sqexp_exp[0],params_optimal_sqexp_exp[1])
        K_exp_exp[k,l]=cov_fun_exp_model(t[k],t[l],params_optimal_exp_exp[0],params_optimal_exp_exp[1])
        
K_KI_exp=C_gamma_exp

K_t_sqexp_exp=K_sqexp_exp[:,sample_index.astype(int)]
K_t_exp_exp=K_exp_exp[:,sample_index.astype(int)]
K_t_Bochner_exp=K_Bochner_exp[:,sample_index.astype(int)]
K_t_KI_exp=K_KI_exp[:,sample_index.astype(int)]

K_ij_sqexp_exp=K_sqexp_exp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_exp_exp=K_exp_exp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_Bochner_exp=K_Bochner_exp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_KI_exp=K_KI_exp[np.ix_(sample_index.astype(int),sample_index.astype(int))]


# Third: bb Dataset

K_sqexp_bb=np.zeros([n,n])
K_exp_bb=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_bb[k,l]=cov_fun_sqexp_model(t[k],t[l],params_optimal_sqexp_bb[0],params_optimal_sqexp_bb[1])
        K_exp_bb[k,l]=cov_fun_exp_model(t[k],t[l],params_optimal_exp_bb[0],params_optimal_exp_bb[1])
        
K_KI_bb=C_gamma_bb

K_t_sqexp_bb=K_sqexp_bb[:,sample_index.astype(int)]
K_t_exp_bb=K_exp_bb[:,sample_index.astype(int)]
K_t_Bochner_bb=K_Bochner_bb[:,sample_index.astype(int)]
K_t_KI_bb=K_KI_bb[:,sample_index.astype(int)]

K_ij_sqexp_bb=K_sqexp_bb[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_exp_bb=K_exp_bb[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_Bochner_bb=K_Bochner_bb[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_KI_bb=K_KI_bb[np.ix_(sample_index.astype(int),sample_index.astype(int))]



# Fourth: ndm_1 Dataset

K_sqexp_ndm_1=np.zeros([n,n])
K_exp_ndm_1=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_ndm_1[k,l]=cov_fun_sqexp_model(t[k],t[l],params_optimal_sqexp_ndm_1[0],params_optimal_sqexp_ndm_1[1])
        K_exp_ndm_1[k,l]=cov_fun_exp_model(t[k],t[l],params_optimal_exp_ndm_1[0],params_optimal_exp_ndm_1[1])
        
K_KI_ndm_1=C_gamma_ndm_1

K_t_sqexp_ndm_1=K_sqexp_ndm_1[:,sample_index.astype(int)]
K_t_exp_ndm_1=K_exp_ndm_1[:,sample_index.astype(int)]
K_t_Bochner_ndm_1=K_Bochner_ndm_1[:,sample_index.astype(int)]
K_t_KI_ndm_1=K_KI_ndm_1[:,sample_index.astype(int)]

K_ij_sqexp_ndm_1=K_sqexp_ndm_1[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_exp_ndm_1=K_exp_ndm_1[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_Bochner_ndm_1=K_Bochner_ndm_1[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_KI_ndm_1=K_KI_ndm_1[np.ix_(sample_index.astype(int),sample_index.astype(int))]



# Fifth: ndm_2 Dataset

K_sqexp_ndm_2=np.zeros([n,n])
K_exp_ndm_2=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_ndm_2[k,l]=cov_fun_sqexp_model(t[k],t[l],params_optimal_sqexp_ndm_2[0],params_optimal_sqexp_ndm_2[1])
        K_exp_ndm_2[k,l]=cov_fun_exp_model(t[k],t[l],params_optimal_exp_ndm_2[0],params_optimal_exp_ndm_2[1])
        
K_KI_ndm_2=C_gamma_ndm_2

K_t_sqexp_ndm_2=K_sqexp_ndm_2[:,sample_index.astype(int)]
K_t_exp_ndm_2=K_exp_ndm_2[:,sample_index.astype(int)]
K_t_Bochner_ndm_2=K_Bochner_ndm_2[:,sample_index.astype(int)]
K_t_KI_ndm_2=K_KI_ndm_2[:,sample_index.astype(int)]

K_ij_sqexp_ndm_2=K_sqexp_ndm_2[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_exp_ndm_2=K_exp_ndm_2[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_Bochner_ndm_2=K_Bochner_ndm_2[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_KI_ndm_2=K_KI_ndm_2[np.ix_(sample_index.astype(int),sample_index.astype(int))]


# Sixth: True underlying covariances

K_t_true_sqexp=K_sqexp[:,sample_index.astype(int)]
K_t_true_exp=K_exp[:,sample_index.astype(int)]
K_t_true_bb=K_bb[:,sample_index.astype(int)]
K_t_true_ndm_1=K_nondiag_mercer_1[:,sample_index.astype(int)]
K_t_true_ndm_2=K_nondiag_mercer_2[:,sample_index.astype(int)]

K_ij_true_sqexp=K_sqexp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_true_exp=K_KI_exp[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_true_bb=K_bb[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_true_ndm_1=K_nondiag_mercer_1[np.ix_(sample_index.astype(int),sample_index.astype(int))]
K_ij_true_ndm_2=K_nondiag_mercer_2[np.ix_(sample_index.astype(int),sample_index.astype(int))]



# ii) Perform estimation

# First: sqexp dataset
invtol=10**(-3) 
sqn=np.sqrt(n)

RMSE_sqexp_sqexp=np.zeros([n_test,1])
RMSE_exp_sqexp=np.zeros([n_test,1])
RMSE_Bochner_sqexp=np.zeros([n_test,1])
RMSE_KI_sqexp=np.zeros([n_test,1])
RMSE_true_sqexp=np.zeros([n_test,1])


for k in range(n_test):
    x_temp_sqexp=np.random.multivariate_normal(np.zeros([n]),K_sqexp)
    x_temp_meas_sqexp=x_temp_sqexp[sample_index.astype(int)]
    
    x_temp_sqexp_sqexp=K_t_sqexp_sqexp@lina.pinv(K_ij_sqexp_sqexp,rcond=invtol)@x_temp_meas_sqexp
    x_temp_exp_sqexp=K_t_exp_sqexp@lina.pinv(K_ij_exp_sqexp,rcond=invtol)@x_temp_meas_sqexp
    x_temp_Bochner_sqexp=K_t_Bochner_sqexp@lina.pinv(K_ij_Bochner_sqexp,rcond=invtol)@x_temp_meas_sqexp
    x_temp_KI_sqexp=K_t_KI_sqexp@lina.pinv(K_ij_KI_sqexp,rcond=invtol)@x_temp_meas_sqexp
    x_temp_true_sqexp=K_t_true_sqexp@lina.pinv(K_ij_true_sqexp,rcond=invtol)@x_temp_meas_sqexp
    
    
    RMSE_sqexp_sqexp[k]=lina.norm(x_temp_sqexp-x_temp_sqexp_sqexp)
    RMSE_exp_sqexp[k]=lina.norm(x_temp_sqexp-x_temp_exp_sqexp)
    RMSE_Bochner_sqexp[k]=lina.norm(x_temp_sqexp-x_temp_Bochner_sqexp)
    RMSE_KI_sqexp[k]=lina.norm(x_temp_sqexp-x_temp_KI_sqexp)
    RMSE_true_sqexp[k]=lina.norm(x_temp_sqexp-x_temp_true_sqexp)
    
RMSE_sqexp_sqexp_mean=np.mean(RMSE_sqexp_sqexp)/sqn
RMSE_exp_sqexp_mean=np.mean(RMSE_exp_sqexp)/sqn
RMSE_Bochner_sqexp_mean=np.mean(RMSE_Bochner_sqexp)/sqn
RMSE_KI_sqexp_mean=np.mean(RMSE_KI_sqexp)/sqn
RMSE_true_sqexp_mean=np.mean(RMSE_true_sqexp)/sqn

print('Sqexp simulations done!')
    

# Second: exp dataset

RMSE_sqexp_exp=np.zeros([n_test,1])
RMSE_exp_exp=np.zeros([n_test,1])
RMSE_Bochner_exp=np.zeros([n_test,1])
RMSE_KI_exp=np.zeros([n_test,1])
RMSE_true_exp=np.zeros([n_test,1])


for k in range(n_test):
    x_temp_exp=np.random.multivariate_normal(np.zeros([n]),K_exp)
    x_temp_meas_exp=x_temp_exp[sample_index.astype(int)]
    
    x_temp_sqexp_exp=K_t_sqexp_exp@lina.pinv(K_ij_sqexp_exp,rcond=invtol)@x_temp_meas_exp
    x_temp_exp_exp=K_t_exp_exp@lina.pinv(K_ij_exp_exp,rcond=invtol)@x_temp_meas_exp
    x_temp_Bochner_exp=K_t_Bochner_exp@lina.pinv(K_ij_Bochner_exp,rcond=invtol)@x_temp_meas_exp
    x_temp_KI_exp=K_t_KI_exp@lina.pinv(K_ij_KI_exp,rcond=invtol)@x_temp_meas_exp
    x_temp_true_exp=K_t_true_exp@lina.pinv(K_ij_true_exp,rcond=invtol)@x_temp_meas_exp
    
    
    RMSE_sqexp_exp[k]=lina.norm(x_temp_exp-x_temp_sqexp_exp)/sqn
    RMSE_exp_exp[k]=lina.norm(x_temp_exp-x_temp_exp_exp)/sqn
    RMSE_Bochner_exp[k]=lina.norm(x_temp_exp-x_temp_Bochner_exp)/sqn
    RMSE_KI_exp[k]=lina.norm(x_temp_exp-x_temp_KI_exp)/sqn
    RMSE_true_exp[k]=lina.norm(x_temp_exp-x_temp_true_exp)
    
RMSE_sqexp_exp_mean=np.mean(RMSE_sqexp_exp)
RMSE_exp_exp_mean=np.mean(RMSE_exp_exp)
RMSE_Bochner_exp_mean=np.mean(RMSE_Bochner_exp)
RMSE_KI_exp_mean=np.mean(RMSE_KI_exp)
RMSE_true_exp_mean=np.mean(RMSE_true_exp)/sqn

print('Exp simulations done!')


# Third: bb dataset


RMSE_sqexp_bb=np.zeros([n_test,1])
RMSE_exp_bb=np.zeros([n_test,1])
RMSE_Bochner_bb=np.zeros([n_test,1])
RMSE_KI_bb=np.zeros([n_test,1])
RMSE_true_bb=np.zeros([n_test,1])


for k in range(n_test):
    x_temp_bb=np.random.multivariate_normal(np.zeros([n]),K_bb)
    x_temp_meas_bb=x_temp_bb[sample_index.astype(int)]
    
    x_temp_sqexp_bb=K_t_sqexp_bb@lina.pinv(K_ij_sqexp_bb,rcond=invtol)@x_temp_meas_bb
    x_temp_exp_bb=K_t_exp_bb@lina.pinv(K_ij_exp_bb,rcond=invtol)@x_temp_meas_bb
    x_temp_Bochner_bb=K_t_Bochner_bb@lina.pinv(K_ij_Bochner_bb,rcond=invtol)@x_temp_meas_bb
    x_temp_KI_bb=K_t_KI_bb@lina.pinv(K_ij_KI_bb,rcond=invtol)@x_temp_meas_bb
    x_temp_true_bb=K_t_true_bb@lina.pinv(K_ij_true_bb,rcond=invtol)@x_temp_meas_bb
    
    RMSE_sqexp_bb[k]=lina.norm(x_temp_bb-x_temp_sqexp_bb)/sqn
    RMSE_exp_bb[k]=lina.norm(x_temp_bb-x_temp_exp_bb)/sqn
    RMSE_Bochner_bb[k]=lina.norm(x_temp_bb-x_temp_Bochner_bb)/sqn
    RMSE_KI_bb[k]=lina.norm(x_temp_bb-x_temp_KI_bb)/sqn
    RMSE_true_bb[k]=lina.norm(x_temp_bb-x_temp_true_bb)
    
RMSE_sqexp_bb_mean=np.mean(RMSE_sqexp_bb)
RMSE_exp_bb_mean=np.mean(RMSE_exp_bb)
RMSE_Bochner_bb_mean=np.mean(RMSE_Bochner_bb)
RMSE_KI_bb_mean=np.mean(RMSE_KI_bb)
RMSE_true_bb_mean=np.mean(RMSE_true_bb)/sqn

print('Brownian bridge simulations done!')



# Fourth: ndm_1 dataset

RMSE_sqexp_ndm_1=np.zeros([n_test,1])
RMSE_exp_ndm_1=np.zeros([n_test,1])
RMSE_Bochner_ndm_1=np.zeros([n_test,1])
RMSE_KI_ndm_1=np.zeros([n_test,1])
RMSE_true_ndm_1=np.zeros([n_test,1])


for k in range(n_test):
    x_temp_ndm_1=np.random.multivariate_normal(np.zeros([n]),K_nondiag_mercer_1)
    x_temp_meas_ndm_1=x_temp_ndm_1[sample_index.astype(int)]
    
    x_temp_sqexp_ndm_1=K_t_sqexp_ndm_1@lina.pinv(K_ij_sqexp_ndm_1,rcond=invtol)@x_temp_meas_ndm_1
    x_temp_exp_ndm_1=K_t_exp_ndm_1@lina.pinv(K_ij_exp_ndm_1,rcond=invtol)@x_temp_meas_ndm_1
    x_temp_Bochner_ndm_1=K_t_Bochner_ndm_1@lina.pinv(K_ij_Bochner_ndm_1,rcond=invtol)@x_temp_meas_ndm_1
    x_temp_KI_ndm_1=K_t_KI_ndm_1@lina.pinv(K_ij_KI_ndm_1,rcond=invtol)@x_temp_meas_ndm_1
    x_temp_true_ndm_1=K_t_true_ndm_1@lina.pinv(K_ij_true_ndm_1,rcond=invtol)@x_temp_meas_ndm_1
    
    
    RMSE_sqexp_ndm_1[k]=lina.norm(x_temp_ndm_1-x_temp_sqexp_ndm_1)/sqn
    RMSE_exp_ndm_1[k]=lina.norm(x_temp_ndm_1-x_temp_exp_ndm_1)/sqn
    RMSE_Bochner_ndm_1[k]=lina.norm(x_temp_ndm_1-x_temp_Bochner_ndm_1)/sqn
    RMSE_KI_ndm_1[k]=lina.norm(x_temp_ndm_1-x_temp_KI_ndm_1)/sqn
    RMSE_true_ndm_1[k]=lina.norm(x_temp_ndm_1-x_temp_true_ndm_1)
    
RMSE_sqexp_ndm_1_mean=np.mean(RMSE_sqexp_ndm_1)
RMSE_exp_ndm_1_mean=np.mean(RMSE_exp_ndm_1)
RMSE_Bochner_ndm_1_mean=np.mean(RMSE_Bochner_ndm_1)
RMSE_KI_ndm_1_mean=np.mean(RMSE_KI_ndm_1)
RMSE_true_ndm_1_mean=np.mean(RMSE_true_ndm_1)/sqn

print('Nondiagonal Mercer simulations done!')


# Fifth: ndm_2 dataset

   
RMSE_sqexp_ndm_2=np.zeros([n_test,1])
RMSE_exp_ndm_2=np.zeros([n_test,1])
RMSE_Bochner_ndm_2=np.zeros([n_test,1])
RMSE_KI_ndm_2=np.zeros([n_test,1])
RMSE_true_ndm_2=np.zeros([n_test,1])


for k in range(n_test):
    x_temp_ndm_2=np.random.multivariate_normal(np.zeros([n]),K_nondiag_mercer_2)
    x_temp_meas_ndm_2=x_temp_ndm_2[sample_index.astype(int)]
    
    x_temp_sqexp_ndm_2=K_t_sqexp_ndm_2@lina.pinv(K_ij_sqexp_ndm_2,rcond=invtol)@x_temp_meas_ndm_2
    x_temp_exp_ndm_2=K_t_exp_ndm_2@lina.pinv(K_ij_exp_ndm_2,rcond=invtol)@x_temp_meas_ndm_2
    x_temp_Bochner_ndm_2=K_t_Bochner_ndm_2@lina.pinv(K_ij_Bochner_ndm_2,rcond=invtol)@x_temp_meas_ndm_2
    x_temp_KI_ndm_2=K_t_KI_ndm_2@lina.pinv(K_ij_KI_ndm_2,rcond=invtol)@x_temp_meas_ndm_2
    x_temp_true_ndm_2=K_t_true_ndm_2@lina.pinv(K_ij_true_ndm_2,rcond=invtol)@x_temp_meas_ndm_2
    
    
    RMSE_sqexp_ndm_2[k]=lina.norm(x_temp_ndm_2-x_temp_sqexp_ndm_2)/sqn
    RMSE_exp_ndm_2[k]=lina.norm(x_temp_ndm_2-x_temp_exp_ndm_2)/sqn
    RMSE_Bochner_ndm_2[k]=lina.norm(x_temp_ndm_2-x_temp_Bochner_ndm_2)/sqn
    RMSE_KI_ndm_2[k]=lina.norm(x_temp_ndm_2-x_temp_KI_ndm_2)/sqn
    RMSE_true_ndm_2[k]=lina.norm(x_temp_ndm_2-x_temp_true_ndm_2)
    
RMSE_sqexp_ndm_2_mean=np.mean(RMSE_sqexp_ndm_2)
RMSE_exp_ndm_2_mean=np.mean(RMSE_exp_ndm_2)
RMSE_Bochner_ndm_2_mean=np.mean(RMSE_Bochner_ndm_2)
RMSE_KI_ndm_2_mean=np.mean(RMSE_KI_ndm_2) 
RMSE_true_ndm_2_mean=np.mean(RMSE_true_ndm_2)/sqn

print('Nondiagonal Mercer simulations done!')



"""
    8. Plots and illustrations -----------------------------------------------
"""



# i) Example interpolations

circle_size=200
zero_line=np.zeros([n,1])
zero_line_illu=np.zeros([n,1])

# First row : Sqexp covariance


w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 5)

f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,x_temp_sqexp,linestyle='dashed',color='0.7', label='Ground truth')
f1_ax1.plot(t,x_temp_sqexp_sqexp,linestyle='solid',color='0.0',label='Estimation')
f1_ax1.scatter(t_sample,x_temp_meas_sqexp,facecolors='none',edgecolors='0',label='Observations',s=circle_size)
f1_ax1.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Sqexp cov for sqexp process')
f1_ax1.legend(loc='upper left')

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(t,x_temp_exp,linestyle='dashed',color='0.7')
f1_ax2.plot(t,x_temp_sqexp_exp,linestyle='solid',color='0.0')
f1_ax2.scatter(t_sample,x_temp_meas_exp,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f1_ax2.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Sqexp cov for exp process')

f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.plot(t,x_temp_bb,linestyle='dashed',color='0.7')
f1_ax3.plot(t,x_temp_sqexp_bb,linestyle='solid',color='0.0')
f1_ax3.scatter(t_sample,x_temp_meas_bb,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f1_ax3.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax3.set_title('Sqexp cov for Brownian bridge process')

f1_ax4 = fig1.add_subplot(gs1[0,3])
f1_ax4.plot(t,x_temp_ndm_1,linestyle='dashed',color='0.7')
f1_ax4.plot(t,x_temp_sqexp_ndm_1,linestyle='solid',color='0.0')
f1_ax4.scatter(t_sample,x_temp_meas_ndm_1,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f1_ax4.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax4.set_title('Sqexp cov for random cov process 1')


f1_ax5 = fig1.add_subplot(gs1[0,4])
f1_ax5.plot(t,x_temp_ndm_2,linestyle='dashed',color='0.7')
f1_ax5.plot(t,x_temp_sqexp_ndm_2,linestyle='solid',color='0.0')
f1_ax5.scatter(t_sample,x_temp_meas_ndm_2,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f1_ax5.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax5.set_title('Sqexp cov for random cov process 2')



# Second row : Exp covariance

w,h=plt.figaspect(0.3)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig2.add_gridspec(1, 5)

f2_ax6 = fig2.add_subplot(gs2[0,0])
f2_ax6.plot(t,x_temp_sqexp,linestyle='dashed',color='0.7', label='Ground truth')
f2_ax6.plot(t,x_temp_exp_sqexp,linestyle='solid',color='0.0', label='Estimation')
f2_ax6.scatter(t_sample,x_temp_meas_sqexp,facecolors='none',edgecolors='0',label='Observations',s=circle_size)
f2_ax6.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax6.set_title('Exp cov for sqexp process')
f2_ax6.legend(loc='upper left')


f2_ax7 = fig2.add_subplot(gs2[0,1])
f2_ax7.plot(t,x_temp_exp,linestyle='dashed',color='0.7')
f2_ax7.plot(t,x_temp_exp_exp,linestyle='solid',color='0.0')
f2_ax7.scatter(t_sample,x_temp_meas_exp,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f2_ax7.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax7.set_title('Exp cov for exp process')

f2_ax8 = fig2.add_subplot(gs2[0,2])
f2_ax8.plot(t,x_temp_bb,linestyle='dashed',color='0.7')
f2_ax8.plot(t,x_temp_exp_bb,linestyle='solid',color='0.0')
f2_ax8.scatter(t_sample,x_temp_meas_bb,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f2_ax8.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax8.set_title('Exp cov for Brownian bridge process')

f2_ax9 = fig2.add_subplot(gs2[0,3])
f2_ax9.plot(t,x_temp_ndm_1,linestyle='dashed',color='0.7')
f2_ax9.plot(t,x_temp_exp_ndm_1,linestyle='solid',color='0.0')
f2_ax9.scatter(t_sample,x_temp_meas_ndm_1,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f2_ax9.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax9.set_title('Exp cov for random cov process 1')


f2_ax10 = fig2.add_subplot(gs2[0,4])
f2_ax10.plot(t,x_temp_ndm_2,linestyle='dashed',color='0.7')
f2_ax10.plot(t,x_temp_exp_ndm_2,linestyle='solid',color='0.0')
f2_ax10.scatter(t_sample,x_temp_meas_ndm_2,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f2_ax10.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax10.set_title('Exp cov for random cov process 2')



# Third row : Bochner covariance

w,h=plt.figaspect(0.3)
fig3 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs3 = fig3.add_gridspec(1, 5)

f3_ax11 = fig3.add_subplot(gs3[0,0])
f3_ax11.plot(t,x_temp_sqexp,linestyle='dashed',color='0.7', label='Ground truth')
f3_ax11.plot(t,x_temp_Bochner_sqexp,linestyle='solid',color='0.0', label='Estimation')
f3_ax11.scatter(t_sample,x_temp_meas_sqexp,facecolors='none',edgecolors='0',label='Observations',s=circle_size)
f3_ax11.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax11.set_title('Bochner cov for sqexp process')
f3_ax11.legend(loc='upper left')


f3_ax12 = fig3.add_subplot(gs3[0,1])
f3_ax12.plot(t,x_temp_exp,linestyle='dashed',color='0.7')
f3_ax12.plot(t,x_temp_Bochner_exp,linestyle='solid',color='0.0')
f3_ax12.scatter(t_sample,x_temp_meas_exp,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f3_ax12.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax12.set_title('Bochner cov for exp process')

f3_ax13 = fig3.add_subplot(gs3[0,2])
f3_ax13.plot(t,x_temp_bb,linestyle='dashed',color='0.7')
f3_ax13.plot(t,x_temp_Bochner_bb,linestyle='solid',color='0.0')
f3_ax13.scatter(t_sample,x_temp_meas_bb,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f3_ax13.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax13.set_title('Bochner cov for Brownian bridge process')

f3_ax14 = fig3.add_subplot(gs3[0,3])
f3_ax14.plot(t,x_temp_ndm_1,linestyle='dashed',color='0.7')
f3_ax14.plot(t,x_temp_Bochner_ndm_1,linestyle='solid',color='0.0')
f3_ax14.scatter(t_sample,x_temp_meas_ndm_1,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f3_ax14.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax14.set_title('Bochner cov for random cov process 1')


f3_ax15 = fig3.add_subplot(gs3[0,4])
f3_ax15.plot(t,x_temp_ndm_2,linestyle='dashed',color='0.7')
f3_ax15.plot(t,x_temp_Bochner_ndm_2,linestyle='solid',color='0.0')
f3_ax15.scatter(t_sample,x_temp_meas_ndm_2,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f3_ax15.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax15.set_title('Bochner cov for random cov process 2')



# Fourth row : KI covariance

w,h=plt.figaspect(0.3)
fig4 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs4 = fig4.add_gridspec(1, 5)

f4_ax16 = fig4.add_subplot(gs4[0,0])
f4_ax16.plot(t,x_temp_sqexp,linestyle='dashed',color='0.7',label='Ground truth')
f4_ax16.plot(t,x_temp_KI_sqexp,linestyle='solid',color='0.0', label='Estimation')
f4_ax16.scatter(t_sample,x_temp_meas_sqexp,facecolors='none',edgecolors='0',label='Observations',s=circle_size)
f4_ax16.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f4_ax16.set_title('KI cov for sqexp process')
f4_ax16.legend(loc='upper left')


f4_ax17 = fig4.add_subplot(gs4[0,1])
f4_ax17.plot(t,x_temp_exp,linestyle='dashed',color='0.7')
f4_ax17.plot(t,x_temp_KI_exp,linestyle='solid',color='0.0')
f4_ax17.scatter(t_sample,x_temp_meas_exp,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f4_ax17.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f4_ax17.set_title('KI cov for exp process')

f4_ax18 = fig4.add_subplot(gs4[0,2])
f4_ax18.plot(t,x_temp_bb,linestyle='dashed',color='0.7')
f4_ax18.plot(t,x_temp_KI_bb,linestyle='solid',color='0.0')
f4_ax18.scatter(t_sample,x_temp_meas_bb,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f4_ax18.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f4_ax18.set_title('KI cov for Brownian bridge process')

f4_ax19 = fig4.add_subplot(gs4[0,3])
f4_ax19.plot(t,x_temp_ndm_1,linestyle='dashed',color='0.7')
f4_ax19.plot(t,x_temp_KI_ndm_1,linestyle='solid',color='0.0')
f4_ax19.scatter(t_sample,x_temp_meas_ndm_1,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f4_ax19.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f4_ax19.set_title('KI cov for random cov process 1')


f4_ax20 = fig4.add_subplot(gs4[0,4])
f4_ax20.plot(t,x_temp_ndm_2,linestyle='dashed',color='0.7')
f4_ax20.plot(t,x_temp_KI_ndm_2,linestyle='solid',color='0.0')
f4_ax20.scatter(t_sample,x_temp_meas_ndm_2,facecolors='none',edgecolors='0',label='Noise',s=circle_size)
f4_ax20.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f4_ax20.set_title('KI cov for random cov process 2')




# ii) Example covariance fits


# Calculations for ground truth

circle_size=100

cov_true_sqexp=np.zeros([n])
cov_true_exp=np.zeros([n])
cov_true_bb=np.zeros([n])

for k in range(n):
    cov_true_sqexp[k]=cov_fun_sqexp_true(0, t[k])   
    cov_true_exp[k]=cov_fun_exp_true(0, t[k])
    cov_true_bb[k]=cov_fun_bb_true(0, t[k])
    
    
    
cov_sqexp_sqexp=np.zeros([n])
cov_sqexp_exp=np.zeros([n])
cov_sqexp_bb=np.zeros([n])
cov_sqexp_ndm_1=np.zeros([n])
cov_sqexp_ndm_2=np.zeros([n])


cov_exp_sqexp=np.zeros([n])
cov_exp_exp=np.zeros([n])
cov_exp_bb=np.zeros([n])
cov_exp_ndm_1=np.zeros([n])
cov_exp_ndm_2=np.zeros([n])

cov_Bochner_sqexp=np.zeros([n])
cov_Bochner_exp=np.zeros([n])
cov_Bochner_bb=np.zeros([n])
cov_Bochner_ndm_1=np.zeros([n])
cov_Bochner_ndm_2=np.zeros([n])

    
for k in range(n):
    cov_sqexp_sqexp[k]=cov_fun_sqexp_model(t[k],0,params_optimal_sqexp_exp[0],params_optimal_sqexp_sqexp[1])
    cov_sqexp_exp[k]=cov_fun_sqexp_model(t[k],0,params_optimal_sqexp_exp[0],params_optimal_sqexp_exp[1])
    cov_sqexp_bb[k]=cov_fun_sqexp_model(t[k],0,params_optimal_sqexp_bb[0],params_optimal_sqexp_bb[1])
    cov_sqexp_ndm_1[k]=cov_fun_sqexp_model(t[k],0,params_optimal_sqexp_ndm_1[0],params_optimal_sqexp_ndm_1[1])
    cov_sqexp_ndm_2[k]=cov_fun_sqexp_model(t[k],0,params_optimal_sqexp_ndm_2[0],params_optimal_sqexp_ndm_2[1])
    
    cov_exp_sqexp[k]=cov_fun_exp_model(t[k],0,params_optimal_exp_sqexp[0],params_optimal_exp_sqexp[1])
    cov_exp_exp[k]=cov_fun_exp_model(t[k],0,params_optimal_exp_exp[0],params_optimal_exp_exp[1])
    cov_exp_bb[k]=cov_fun_exp_model(t[k],0,params_optimal_exp_bb[0],params_optimal_exp_bb[1])
    cov_exp_ndm_1[k]=cov_fun_exp_model(t[k],0,params_optimal_exp_ndm_1[0],params_optimal_exp_ndm_1[1])
    cov_exp_ndm_2[k]=cov_fun_exp_model(t[k],0,params_optimal_exp_ndm_2[0],params_optimal_exp_ndm_2[1])
    
cov_Bochner_sqexp=K_Bochner_sqexp[:,0]
cov_Bochner_exp=K_Bochner_exp[:,0]
cov_Bochner_bb=K_Bochner_bb[:,0]
cov_Bochner_ndm_1=K_Bochner_ndm_1[:,0]
cov_Bochner_ndm_2=K_Bochner_ndm_2[:,0]


 # First row : Sqexp covariance


w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)

f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,cov_true_sqexp,linestyle='dashed',color='0.7', label='True covariance')
f1_ax1.plot(t,cov_sqexp_sqexp,linestyle='solid',color='0.0',label='Estimation')
f1_ax1.plot(t,correlogram_sqexp,linestyle='dotted',color='0.0',label='Correlogram')
f1_ax1.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Sqexp cov fun for sqexp process')
f1_ax1.legend(loc='upper left')

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(t,cov_true_exp,linestyle='dashed',color='0.7', label='True covariance')
f1_ax2.plot(t,cov_sqexp_exp,linestyle='solid',color='0.0',label='Estimation')
f1_ax2.plot(t,correlogram_exp,linestyle='dotted',color='0.0',label='Correlogram')
f1_ax2.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Sqexp cov fun for exp process')

f1_ax3 = fig1.add_subplot(gs1[0,2])
#f1_ax3.plot(t,cov_true_bb,linestyle='dashed',color='0.7', label='True covariance')
f1_ax3.plot(t,cov_sqexp_bb,linestyle='solid',color='0.0',label='Estimation')
f1_ax3.plot(t,correlogram_bb,linestyle='dotted',color='0.0',label='Correlogram')
f1_ax3.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax3.set_title('Sqexp cov fun for Brownian bridge process')




# Second row : Exp covariance

w,h=plt.figaspect(0.3)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig2.add_gridspec(1, 3)

f2_ax1 = fig2.add_subplot(gs2[0,0])
f2_ax1.plot(t,cov_true_sqexp,linestyle='dashed',color='0.7', label='True covariance')
f2_ax1.plot(t,cov_exp_sqexp,linestyle='solid',color='0.0',label='Estimation')
f2_ax1.plot(t,correlogram_sqexp,linestyle='dotted',color='0.0',label='Correlogram')
f2_ax1.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax1.set_title('Exp cov fun for sqexp process')
f2_ax1.legend(loc='upper left')

f2_ax2 = fig2.add_subplot(gs2[0,1])
f2_ax2.plot(t,cov_true_exp,linestyle='dashed',color='0.7', label='True covariance')
f2_ax2.plot(t,cov_exp_exp,linestyle='solid',color='0.0',label='Estimation')
f2_ax2.plot(t,correlogram_exp,linestyle='dotted',color='0.0',label='Correlogram')
f2_ax2.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax2.set_title('Exp cov fun for exp process')

f2_ax3 = fig2.add_subplot(gs2[0,2])
#f2_ax3.plot(t,cov_true_bb,linestyle='dashed',color='0.7', label='True covariance')
f2_ax3.plot(t,cov_exp_bb,linestyle='solid',color='0.0',label='Estimation')
f2_ax3.plot(t,correlogram_bb,linestyle='dotted',color='0.0',label='Correlogram')
f2_ax3.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax3.set_title('Exp cov fun for Brownian bridge process')


# Third row : Bochner covariance

w,h=plt.figaspect(0.3)
fig3 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs3 = fig3.add_gridspec(1, 3)

f3_ax1 = fig3.add_subplot(gs3[0,0])
f3_ax1.plot(t,cov_true_sqexp,linestyle='dashed',color='0.7', label='True covariance')
f3_ax1.plot(t,cov_Bochner_sqexp,linestyle='solid',color='0.0',label='Estimation')
f3_ax1.plot(t,correlogram_sqexp,linestyle='dotted',color='0.0',label='Correlogram')
f3_ax1.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax1.set_title('Bochner cov fun for sqexp process')
f3_ax1.legend(loc='upper left')

f3_ax2 = fig3.add_subplot(gs3[0,1])
f3_ax2.plot(t,cov_true_exp,linestyle='dashed',color='0.7', label='True covariance')
f3_ax2.plot(t,cov_Bochner_exp,linestyle='solid',color='0.0',label='Estimation')
f3_ax2.plot(t,correlogram_exp,linestyle='dotted',color='0.0',label='Correlogram')
f3_ax2.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax2.set_title('Bochner cov fun for exp process')

f3_ax3 = fig3.add_subplot(gs3[0,2])
#f3_ax3.plot(t,cov_true_bb,linestyle='dashed',color='0.7', label='True covariance')
f3_ax3.plot(t,cov_Bochner_bb,linestyle='solid',color='0.0',label='Estimation')
f3_ax3.plot(t,correlogram_bb,linestyle='dotted',color='0.0',label='Correlogram')
f3_ax3.plot(t,zero_line,linestyle='dotted',color='0.75')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f3_ax3.set_title('Bochner cov fun for Brownian bridge process')























