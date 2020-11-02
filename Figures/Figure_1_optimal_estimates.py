"""
The goal of this script is to showcase mean-square optimal estimation using 
different covariance functions and in this way produce figure 1 of the paper
'Inference of instationary covariance functions for optimal estimation in 
spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Simulation of autocorrelated data
    3. Optimal estimation 
    4. Plots and illustrations
    
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
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


# ii) Definition of auxiliary quantities

n=300
n_sample=5

t=np.linspace(0,1,n)
random_sample_index=np.sort(np.random.choice(range(n),size=n_sample))
t_sample=t[random_sample_index]
np.random.seed(1)




"""
    2. Simulation of autocorrelated data -------------------------------------
"""


# i) Define true underlying covariance function

def cov_fun_true(t1,t2):
    return np.min([t1,t2])-t1*t2


# ii) Set up covariance matrix

K_true=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_true[k,l]=cov_fun_true(t[k],t[l])


# iii) Generate simulations

mu_true=np.reshape(np.zeros([n,1]),[n])

x_simu=np.random.multivariate_normal(mu_true,K_true) 
x_sample=x_simu[random_sample_index.astype(int)]

S_emp=x_simu@x_simu.T
S_sample=x_sample@x_sample.T



"""
    3. Optimal estimation  ---------------------------------------------------
"""


# i) Define covariance models

d_sqexp=0.1
d_exp=0.2

def cov_fun_sqexp(t1,t2):
    return 1*np.exp(-(lina.norm(t1-t2)/d_sqexp)**2)

def cov_fun_exp(t1,t2):
    return 1*np.exp(-(lina.norm(t1-t2)/d_exp)**1)


# ii) Generate full covariance matrices

K_sqexp_full=np.zeros([n,n])
K_exp_full=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        K_sqexp_full[k,l]=cov_fun_sqexp(t[k],t[l])    
        K_exp_full[k,l]=cov_fun_exp(t[k],t[l])
        
        
# iii) Generate covariance matrices for estimation

K_true_sample=np.zeros([n_sample,n_sample])
K_sqexp_sample=np.zeros([n_sample,n_sample])
K_exp_sample=np.zeros([n_sample,n_sample])

for k in range(n_sample):
    for l in range(n_sample):
        K_true_sample[k,l]=cov_fun_true(t_sample[k],t_sample[l])
        K_sqexp_sample[k,l]=cov_fun_sqexp(t_sample[k],t_sample[l])    
        K_exp_sample[k,l]=cov_fun_exp(t_sample[k],t_sample[l])


K_true_subset=K_true[:,random_sample_index]
K_sqexp_subset=K_sqexp_full[:,random_sample_index]
K_exp_subset=K_exp_full[:,random_sample_index]        
        
        
# iv) Perform estimation

x_est_true_cov=K_true_subset@lina.pinv(K_true_sample)@x_sample
x_est_sqexp_cov=K_sqexp_subset@lina.pinv(K_sqexp_sample)@x_sample
x_est_exp_cov=K_exp_subset@lina.pinv(K_exp_sample)@x_sample



"""
    4. Plots and illustrations -----------------------------------------------
"""

# i) Calculate some exemplary covariance values

n_illu=300
t_cov_illu=np.linspace(0,0.5,n_illu)
zero_line=np.zeros([n,1])
zero_line_illu=np.zeros([n_illu,1])

cov_values_sqexp=np.zeros([n_illu,1])
cov_values_exp=np.zeros([n_illu,1])

for k in range(n_illu):
    cov_values_sqexp[k]=cov_fun_sqexp(0,t_cov_illu[k])
    cov_values_exp[k]=cov_fun_exp(0,t_cov_illu[k])
    
    
# ii) Generate plots

w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Truth and optimal estimate
f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,x_simu,linestyle='dotted',color='0.75',label='Underlying truth')
f1_ax1.plot(t,x_est_true_cov,linestyle='dashed',color='0',label='Optimal estimate')
f1_ax1.plot(t,zero_line,linestyle='dotted',color='0.5')
f1_ax1.scatter(t_sample,x_sample,facecolors='none',edgecolors='0',label='Data points')

y_min,y_max=plt.ylim()
y_max=y_max+0.2
plt.ylim(y_min,y_max)
plt.xlabel('Time t')
plt.ylabel('Function value')
f1_ax1.set_title('Truth and optimal estimate')
f1_ax1.legend(loc='upper right')


# Location 1,2 Estimates featuring different covariance functions
f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.scatter(t_sample,x_sample,facecolors='none',edgecolors='0',label='Data points')
f1_ax2.plot(t,x_est_sqexp_cov,linestyle='solid',color='0',label='Estimate sqexp')
f1_ax2.plot(t,x_est_exp_cov,linestyle='dashed',color='0',label='Estimate exp')
f1_ax2.plot(t,zero_line,linestyle='dotted',color='0.5')

plt.ylim(y_min,y_max)
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
f1_ax2.set_title('Other estimates')
f1_ax2.legend(loc='upper right')


# Location 1,3 Plot of the covariance functions
f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.plot(t_cov_illu,cov_values_sqexp,linestyle='solid',color='0',label='Sq exp cov function')
f1_ax3.plot(t_cov_illu,cov_values_exp,linestyle='dashed',color='0',label='Exp cov function')
f1_ax3.plot(t_cov_illu,zero_line_illu,linestyle='dotted',color='0.5')

plt.ylim(-0.1,1.1)
plt.xlabel('Time difference s-t')
plt.ylabel('Function value')
f1_ax3.set_title('Covariance functions')
f1_ax3.legend(loc='upper right')

# Save the figure
# plt.savefig('Figure_1',dpi=400)















