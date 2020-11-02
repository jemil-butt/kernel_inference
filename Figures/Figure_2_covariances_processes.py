"""
The goal of this script is to showcase different covariance matrices arising 
from different model choices together with example realizations of the 
associated stochastic processes. In this way, produce figure 2 of the paper
'Inference of instationary covariance functions for optimal estimation in 
spatial statistics'.

For this, do the following:
    1. Imports and definitions
    2. Create covariance matrices
    3. Sample from stochastic processes
    4. Plots and illustrations
    
The simulations are based on a fixed random seed, to generate data deviating 
from the ones shown in the paper and different for each run, please comment out
the entry 'np.random.seed(x)' in section 1.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import numpy.linalg as lina
from scipy.stats import wishart
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})


# ii) Definition of auxiliary quantities

n=300
t=np.linspace(0,1,n)
np.random.seed(1)



"""
    2. Create covariance matrices  --------------------------------------------
"""


# i) Squared exponential covariance matrices
d_1=0.4; d_2=0.1; 

def cov_fun_sqexp_1(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_1)**2)
    
def cov_fun_sqexp_2(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_2)**2)
    
K_sqexp_1=np.zeros([n,n])
K_sqexp_2=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_sqexp_1[k,l]=cov_fun_sqexp_1(t[k],t[l])
        K_sqexp_2[k,l]=cov_fun_sqexp_2(t[k],t[l])
       

# ii) Bochner covariance functions

n_exp=10

[U_1,Lambda_1,V_1]=lina.svd(K_sqexp_1)
Lambda_1=np.diag(Lambda_1)
Lambda_cut_1=Lambda_1[:n_exp,:n_exp]
U_1_cut=U_1[:,:n_exp]

[U_2,Lambda_2,V_2]=lina.svd(K_sqexp_2)
Lambda_2=np.diag(Lambda_2)
Lambda_cut_2=Lambda_2[:n_exp,:n_exp]
U_2_cut=U_2[:,:n_exp]

weight_fun_1=np.diag(np.diag((1/n_exp)*wishart.rvs(n_exp,scale=Lambda_cut_1)))
weight_fun_2=np.diag(np.diag((1/n_exp)*wishart.rvs(n_exp,scale=Lambda_cut_2)))

omega=np.logspace(-1,1,n_exp)

def complex_exp(t1,t2,omega):
    return np.exp(2*np.pi*(1j)*omega*(t1-t2))

basis_vectors=np.zeros([n,n_exp])+0j*np.zeros([n,n_exp])
for k in range(n):
    for l in range(n_exp):
        basis_vectors[k,l]=(complex_exp(t[k],0,omega[l]))
        
K_Bochner_1=np.real(basis_vectors@weight_fun_1@basis_vectors.conj().T)
K_Bochner_2=np.real(basis_vectors@weight_fun_2@basis_vectors.conj().T)  


# iii) Nondiagonal Mercer covariance 

gamma_1=(1/n_exp)*wishart.rvs(n_exp,scale=Lambda_cut_1)
gamma_2=(1/n_exp)*wishart.rvs(n_exp,scale=Lambda_cut_2)


K_nondiag_Mercer_1=U_1_cut@gamma_1@U_1_cut.T
K_nondiag_Mercer_2=U_2_cut@gamma_2@U_2_cut.T



"""
    3. Sample from stochastic processes --------------------------------------
"""

# i) Prepare simulation
n_simu=10

x_sqexp_1=np.zeros([n,n_simu])
x_sqexp_2=np.zeros([n,n_simu])
x_Bochner_1=np.zeros([n,n_simu])
x_Bochner_2=np.zeros([n,n_simu])
x_nondiag_Mercer_1=np.zeros([n,n_simu])
x_nondiag_Mercer_2=np.zeros([n,n_simu])

for k in range(n_simu):
    x_sqexp_1[:,k]=np.random.multivariate_normal(np.zeros([n]),K_sqexp_1)
    x_sqexp_2[:,k]=np.random.multivariate_normal(np.zeros([n]),K_sqexp_2)
    x_Bochner_1[:,k]=np.random.multivariate_normal(np.zeros([n]),K_Bochner_1)
    x_Bochner_2[:,k]=np.random.multivariate_normal(np.zeros([n]),K_Bochner_2)
    x_nondiag_Mercer_1[:,k]=np.random.multivariate_normal(np.zeros([n]),K_nondiag_Mercer_1)
    x_nondiag_Mercer_2[:,k]=np.random.multivariate_normal(np.zeros([n]),K_nondiag_Mercer_2)




"""
    3. Plots and illustrations -----------------------------------------------
"""

# i) Invoke figure
fig1 = plt.figure(dpi=200,constrained_layout=True)
gs1 = fig1.add_gridspec(3, 4)


# ii) Squared exponential covariance matrices
f1_ax1 = fig1.add_subplot(gs1[0, 0])
f1_ax1.imshow(K_sqexp_1)
f1_ax1.set_title('Squared exponential covariance 1')
f1_ax1.axis('off')

f1_ax2 = fig1.add_subplot(gs1[0, 2])
f1_ax2.imshow(K_sqexp_2)
f1_ax2.set_title('Squared exponential covariance 2')
f1_ax2.axis('off')


# iii) Bochner covariance matrices

f1_ax3 = fig1.add_subplot(gs1[1, 0])
f1_ax3.imshow(K_Bochner_1)
f1_ax3.set_title('Bochner covariance 1')
f1_ax3.axis('off')

f1_ax4 = fig1.add_subplot(gs1[1, 2])
f1_ax4.imshow(K_Bochner_2)
f1_ax4.set_title('Bochner covariance 2')
f1_ax4.axis('off')


# iv) Nondiagonal Mercer covariance matrices

f1_ax5 = fig1.add_subplot(gs1[2, 0])
f1_ax5.imshow(K_nondiag_Mercer_1)
f1_ax5.set_title('Nondiagonal Mercer covariance 1')
f1_ax5.axis('off')

f1_ax6 = fig1.add_subplot(gs1[2, 2])
f1_ax6.imshow(K_nondiag_Mercer_2)
f1_ax6.set_title('Nondiagonal Mercer covariance 2')
f1_ax6.axis('off')


# v) Squared exponential realizations

f1_ax7 = fig1.add_subplot(gs1[0, 1])
f1_ax7.plot(t,x_sqexp_1,color='0')
f1_ax7.set_title('Example realizations')
f1_ax7.axis('off')


f1_ax8 = fig1.add_subplot(gs1[0, 3])
f1_ax8.plot(t,x_sqexp_2,color='0')
f1_ax7.set_title('Example realizations')
f1_ax8.axis('off')


# vi) Bochner realizations

f1_ax7 = fig1.add_subplot(gs1[1, 1])
f1_ax7.plot(t,x_Bochner_1,color='0')
f1_ax7.axis('off')


f1_ax8 = fig1.add_subplot(gs1[1, 3])
f1_ax8.plot(t,x_Bochner_2,color='0')
f1_ax8.axis('off')

# vi) Nondiagonal mercer realizations

f1_ax7 = fig1.add_subplot(gs1[2, 1])
f1_ax7.plot(t,x_nondiag_Mercer_1,color='0')
f1_ax7.axis('off')


f1_ax8 = fig1.add_subplot(gs1[2, 3])
f1_ax8.plot(t,x_nondiag_Mercer_2,color='0')
f1_ax8.axis('off')





# # Save the figure
# plt.savefig('Figure_2',dpi=400)




































