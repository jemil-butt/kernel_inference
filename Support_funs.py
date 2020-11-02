"""
This file provides several support functions that are available after import.
The functions are:
    Logpdet: Calculates the log pseudodeterminant of a matrix
    Backtracking_linesearch: Performs backtracking line search for homogeneous
        kernel inference problems
    Backtracking_linesearch_inhomogeneous: Performs backtracking line search 
        for inhomogeneous kernel inference problems
    Get_P_psi: Calculates an S_psi that satisfies some boundary conditions
    Simulation_random_field: Draws a realization from a random field given
        a tensor product decomposition of its covariance function.
"""




def Logpdet(A,tol):
    """
    The goal of this function is to calculate the log pseudodeterminant of a 
    square matrix A. During the calculation, all eigenvalues smaller than tol
    are set to 1. 
    For this, do the following:
        1. Imports and definitions
        2. Singular value decomposition
        3. Calculate log pseudodeterminant

    INPUTS
    The inputs consist in A square matrix A whose pseudodeterminant is to be 
    calculated and a value for the numerical cutoff-tolerance that determines
    values consideres as zero during the calculation.
    
    Name                 Interpretation                             Type
    A                   Data matrix, each col is one vector-       Matrix [n,n]
                        valued measurement.
    tol                 Tolerance for thresholding the singular    Small positive number
                        values.
                        
                        
    OUTPUTS
    The outputs consist in the pseudodeterminant, a real number
    
    Name                 Interpretation                             Type
    logpdet             The logarithm of the pseudodet             Real number


    """
  
    
    """    
        1. Imports and definitions -------------------------------------------
    """
    
    
    # i) Imports
    
    import numpy as np
    
    
    # ii) Auxiliary quantities
    
    n=np.shape(A)[0]
    
    """        
        2. Singular value decomposition -------------------------------------
    """
    
    
    # i) Calculate and threshold
    
    [U,S,V]=np.linalg.svd(A)
    for k in range(n):
        S[k]=S[k]*(S[k]>tol)+1*(S[k]<=tol)
    
    
    """
        3. Calculate pseudodeterminant ---------------------------------------
    """
    
    
    # i) Calculate pseudodeterminant
    
    logpdet=np.sum(np.log(S))
    
    
    # ii) Assemble solution
    
    return logpdet
    


def Backtracking_linesearch(f, x, lambda_newton, Delta_x,options):
    """
    The goal of this function is to perform a backtracking linesearch to adapt 
    the stepsize t of the Newton step, i.e. prepare a damped Newton step.
    For this, do the following:
        1. Imports and definitions
        2. Loop till conditions satisfied
        
    The stepsize t is reduced until the condition f(x+t Delta_x) < f(x) + 
    t alpha <grad_f, Delta_x> is satisfied.

    INPUTS
    The inputs consist in an objective function f used to check validity of the
    Hessian approximation as well as an evaluation point x and the Newton decrement
    lambda_newton. Furthermore, the descent direction Delta_x needs to be pro-
    vided together with some options on (alpha, beta, tolerances) that feature
    in backtracking line search algorithms.
    
    Name                 Interpretation                             Type
    f                   The objective function for which the       Function handle
                        Armijo optimality condition is to be
                        checked. Calculates the objective values
                        f(x) and f(x+t Delta_x). 
    x                   The position at which gradients and        Matrix [n_exp,n_exp]
                        search directions are evaluated.
    lambda_newton       The Newton decrement quantifying the       A positive real number
                        decrease of the objective function in   
                        the direction of Delta_x           
    Delta_x             Provides the descent direction, for        Matrix [n_exp,n_exp]
                        which a reasonable stepsize t is to be 
                        determined. The recommended update is 
                        then x = x + t Delta x    
    options             Tuple containing the values for alpha,    Tuple (alpha,beta,max_iter) 
                        beta and maximum iterations to arrive 
                        at a reasonable stepsize.
    
    OUTPUTS
    The outputs consist in the stepsize t, a real number guaranteeing that 
    Newton updates do not leave the psd cone.
    
    Name                 Interpretation                             Type
    t                   Stepsize for a robust damped Newton         Real number in [0,1]
                        update                      

    """

    
    
    """
        1. Imports and definitions -------------------------------------------
    """
    
    
    # i) Import packages
    
    import numpy as np
    
    
    # ii) Define auxiliary quantities   
        
    alpha=options[0]
    beta=options[1]
    max_iter=options[2]
    
    # iii) Initial function evaluations
    
    t=1
    f_val_x=f(x)
    f_val_x_mod=f(x+t*Delta_x)
    
    difference=f_val_x_mod-(f_val_x-alpha*t*(lambda_newton**2))
    
    
        
    """    
        2. Loop till conditions satisfied ------------------------------------
    """
    
    # i) Iterate
    
    k=1
    while difference>0 and k<max_iter:
        t=beta*t
        f_val_x_mod=f(x+t*Delta_x)
        difference=f_val_x_mod-(f_val_x-alpha*t*(lambda_newton**2))
        k=k+1
        
    if k==max_iter:
        t=0

    
    # ii) Assemble solution
    
    return t
    
    
    
    
    
def Backtracking_linesearch_inhomogeneous(f, gamma, eta_list, lambda_newton, Delta_gamma, Delta_eta_list, options):
    """
    The goal of this function is to perform a backtracking linesearch to adapt 
    the stepsize t of the Newton step, i.e. prepare a damped Newton step.
    For this, do the following:
        1. Imports and definitions
        2. Loop till conditions satisfied
        
    The stepsize t is reduced until the condition f(x+t Delta_x) < f(x) + 
    t alpha <grad_f, Delta_x> is satisfied.

    INPUTS
    The inputs consist in an objective function f used to check validity of the
    Hessian approximation as well as an evaluation point x and the Newton decrement
    lambda_newton. Furthermore, the descent direction Delta_x needs to be pro-
    vided together with some options on (alpha, beta, tolerances) that feature
    in backtracking line search algorithms.
    
    Name                 Interpretation                             Type
    f                   The objective function for which the       Function handle
                        Armijo optimality condition is to be
                        checked. Calculates the objective values
                        f(x) and f(x+t Delta_x). 
    gamma               The position at which gradients and        Matrix [n_exp,n_exp]
                        search directions are evaluated.
    eta_list            List containing the n_S_obs matrices       List of matrices
                        eta_1 , ... ,eta_{n_S_obs}. Dummy 
                        variables linked to gamma via linear 
                        constraints.
    lambda_newton       The Newton decrement quantifying the       A positive real number
                        decrease of the objective function in   
                        the direction of Delta_x           
    Delta_gamma         Provides the descent direction, for        Matrix [n_exp,n_exp]
                        which a reasonable stepsize t is to be 
                        determined. The recommended update is 
                        then gamma = gamma + t Delta gamma   
    Delta_eta_list      List containing the n_S_obs matrices       List of matrices
                        Delta eta_1 ,..., Delta eta_{n_S_obs}
                        They encode the descent directions for 
                        the eta matrices. The update for the eta 
                        matricesis eta_list[k] = 
                        eta_list[k]+t*Delta_eta_list[k].
    options             Tuple containing the values for alpha,    Tuple (alpha,beta,max_iter) 
                        beta and maximum iterations to arrive 
                        at a reasonable stepsize.
    
    OUTPUTS
    The outputs consist in the stepsize t, a real number guaranteeing that 
    Newton updates do not leave the psd cone.
    
    Name                 Interpretation                             Type
    t                   Stepsize for a robust damped Newton         Real number in [0,1]
     
    """
    
    
    
    """
        1. Imports and definitions -------------------------------------------
    """
    
    
    # i) Import packages
    
    import numpy as np
    
    
    # ii) Define auxiliary quantities   
        
    alpha=options[0]
    beta=options[1]
    max_iter=options[2]
    n_S_obs=len(eta_list)
    
    # iii) Initial function evaluations
    
    t=1
    f_val_x=f(gamma, eta_list)
    
    eta_list_mod=[]
    for k in range(n_S_obs):
        eta_list_mod.append(eta_list[k]+t*Delta_eta_list[k])
    f_val_x_mod=f(gamma+Delta_gamma,eta_list_mod)
    
    difference=f_val_x_mod-(f_val_x-alpha*t*(lambda_newton**2))
    
    
        
    """    
        2. Loop till conditions satisfied ------------------------------------
    """
    
    # i) Iterate
    
    k=1
    while difference>0 and k<max_iter:
        t=beta*t
        
        eta_list_mod=[]
        for k in range(n_S_obs):
            eta_list_mod.append(eta_list[k]+t*Delta_eta_list[k])
        
        f_val_x_mod=f(gamma+t*Delta_gamma,eta_list_mod)
        difference=f_val_x_mod-(f_val_x-alpha*t*(lambda_newton**2))
        k=k+1
        
    if k==max_iter:
        t=0

    
    # ii) Assemble solution
    
    return t
    
    
    
def Get_S_psi(Psi, S_emp, A, b, tol=10**(-6)):
          
    """
    The goal of this function is to provide a matrix S_psi that reconstructs
    the observed empirical covariance matrix S_emp given some constraints that
    are encoded by A vec(S_psi)=b.
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Solve quadratic program
        4. Assemble solutions
        
    INPUTS
    The inputs consist in the matrix Psi used for reconstructing S_emp by 
    Psi@S_psi@Psi.T. This matrix is supposed to be close to the empirical 
    covariance matrix S_emp. The matrix A contains as row vectors the vectorized 
    matrices A_i for which <A_i,gamma>=b_i is supposed to hold.
    
    Name                 Interpretation                             Type
    Psi                 Matrix containing info w.r.t the fun       Matrix [n,n_exp]
                        ction basis used for reconstruction.
                        Each col is one of the basis functions
                        measured by the measurement operator
    S_emp               Empirical covariance matrix to be          Matrix [n,n]
                        approximated via S_psi
    A                   Constraint matrix specifying the linear    Matrix [n_c, n_exp^2]
                        constraints A vec(gamma)=b
    b                   Vector of constraint values                Vector [n_c,1]
    tol                 Tolerance for inversion procedures.        Small positive number
                        The larger the tolerance, the more 
                        regular S_psi is.
                        
    OUTPUTS
    The outputs consist in the matrix S_psi. It reconstructs S_emp closely via
    S_emp approx Psi@S_psi@Psi.T while adhering to the constraints as formulated
    by A.
    
    Name                 Interpretation                             Type
    S_psi               Data induced estimator for gamma,           Matrix [n_exp,n_exp]
                        reconstructs S_emp

    """
    
    
    
    """
        1. Definitions and imports
    """
    
    
    # i) Import packages
    
    import numpy as np
    import numpy.linalg as lina

    
    
    """
        2. Set up problem matrices
    """
    
    
    # i) Gradient and Hessian
    
    F=np.kron(Psi,Psi)
    H=F.T@F
    H_pinv=lina.pinv(H,rcond=tol,hermitian=True)
    
    
    # ii) Respecify dimensions
    
    n=np.shape(Psi)[0]
    n_exp=np.shape(Psi)[1]
    n_c=np.shape(A)[0]
    
    S=np.reshape(S_emp,[n**2,1])
    
    
    
    """
        3. Solve quadratic program
    """
    
    
    # i) Solve QP
    
    # x_1=H_pinv@F.T@S
    # Mid_mat=A.T@(lina.pinv(A@H_pinv@A.T,rcond=tol,hermitian=True))
    # x_2=-H_pinv@Mid_mat@(A@x_1-b)
    
    Top_mat=np.hstack((H,A.T))
    Bot_mat=np.hstack((A,np.zeros([n_c,n_c])))
    
    Full_mat=np.vstack((Top_mat,Bot_mat))
    target_vec=np.vstack((F.T@S,np.zeros([n_c,1])))
    
    # ii) Respecify solution
    
    S_psi=lina.lstsq(Full_mat,target_vec,rcond=tol)[0]
    S_psi=np.reshape(S_psi[:n_exp**2],[n_exp,n_exp])  
    S_psi=0.5*(S_psi+S_psi.T)   
    
    [S,U]=lina.eig(S_psi)
    S_pos=S*(S>0)
    S_psi=U@np.diag(S_pos)@U.T
    
    S_psi=0.5*(S_psi+S_psi.T)
    
    
    
    """
        4. Assemble solutions
    """
    
    
    return S_psi        
    
    
    
     
def Simulation_random_field(cov_x, cov_y, grid_x, grid_y, explained_var):
          
    """
    The goal of this function is to simulate a realization of a random field
    efficiently employing the tensor product nature of covariance functions.
    This does not work for all random field but only for those, whose covariance
    function cov((x_1,y_1),(x_2,y_2)) decomposes as cov_x(x_1,x_2)*cov_y(y_1,y_2). 
    The actual simulation uses the Karhunen Loewe expansion of a process into
    superpositions of basis functions weighted by the eigenvalues of the covariance
    matrix multiplied with white noise variables.
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Simulate and assemble solution
        
    INPUTS
    The inputs consist in the two covariance functions whose product forms the
    multivariate covariance of the random field. Furthermore grid values for
    the input coordinates are provided and a number between 0 and 1 indicating
    how many terms are used in the superposition of the Karhunen Loewe expansion.
    
    Name                 Interpretation                             Type
    cov_x               Function handle for the cov function       function handle
                        Maps two real numbers x_1,x_2 to a real
                        number indicating the cov in x direction
    cov_y               Function handle for the cov function       function handle
                        Maps two real numbers y_1,y_2 to a real
                        number indicating the cov in y direction
    grid_x              Matrix containing info w.r.t the x vals    Matrix [n,n]
                        at each location for which a value is
                        to be simulated
    grid_y              Matrix containing info w.r.t the y vals    Matrix [n,n]
                        at each location for which a value is
                        to be simulated
    explained_var       The fraction of variance to be explained   Number in [0,1]
                        by the simulation. The closer to 1, the 
                        more faithful the reproduction of the cov
                        structure and the longer the runtime
                        
    OUTPUTS
    The outputs consist in the matrix Random_field which is a realization of the
    random field from which a sample was supposed to be drawn.
    
    Name                 Interpretation                             Type
    Random_field        Realization of the random field            Matrix [n,n]


    """
    
    
    
    """
        1. Definitions and imports
    """
    
    
    # i) Import packages
    
    import numpy as np
    import numpy.linalg as lina
    
    
    # ii)) Define auxiliary quantities
    
    n_y,n_x=np.shape(grid_x)
    
    
    
    
    """
        2. Set up problem matrices
    """
    
    
    # i) Component covariance matrices
    
    K_x=np.zeros([n_x,n_x])
    K_y=np.zeros([n_y,n_y])
    for k in range(n_x):
        for l in range(n_x):
            K_x[k,l]=cov_x(grid_x[0,k], grid_x[0,l])
            
    for k in range(n_y):
        for l in range(n_y):
            K_y[k,l]=cov_y(grid_y[k,0], grid_y[l,0])
                
    [U_x,S_x,V_x]=lina.svd(K_x)
    [U_y,S_y,V_y]=lina.svd(K_y)
    

    # ii) Indexing and ordering of eigenvalues
    
    n_tot=n_x*n_y
    
    lambda_mat=np.outer(S_y, S_x)
    index_mat_ordered=np.unravel_index(np.argsort(-lambda_mat.ravel()), [n_y,n_x])
    lambda_ordered=lambda_mat[index_mat_ordered]
    
    lambda_tot=np.sum(lambda_mat)
    lambda_cumsum=np.cumsum(lambda_ordered)
    stop_index=(np.where(lambda_cumsum>=explained_var*lambda_tot))[0][0]
    
        
    
    """
        3. Simulate and assemble solution
    """
        
    
    # i) Iterative Karhunen Loewe composition
    
    white_noise=np.random.normal(0,1,[stop_index])
    
    Random_field=np.zeros([n_y,n_x])
    for k in range(stop_index):
        Random_field=Random_field+white_noise[k]*lambda_ordered[k]*np.outer(U_y[:,index_mat_ordered[0][k]],U_x[:,index_mat_ordered[1][k]])
              
        
        
    # ii) Return solution
    
    return Random_field
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
