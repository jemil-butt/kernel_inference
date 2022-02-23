def Kernel_inference_homogeneous(X,Lambda_p,Psi,r, G="none", A="none", max_iter=100, tol=10**(-2)):
      
    """
    The goal of this function is to provide an algorithm that solves the kernel 
    inference problem in absence of inhomogeneities. It is assumed that 
        i)      the data are all gathered according to one common rule (homogeneous)
        ii)     there is a trend hidden in the data (nonzeromean)
        iii)    constraints on the coefficients gamma are to be imposed (constrained))
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Perform iterations
        4. Assemble solutions
        
    INPUTS
    The inputs consist in a datamatrix X, a prior Lambda on the coefficient tensor gamma,
    and the matrix Psi containing the numerical vectors yielded by applying the measurement
    operator to the sequence of eigenfunctions from the Mercer decomposition of the prior.
    A matrix G provides the link between unknown coefficients beta and the unknown trend
    that is given by G beta. The matrix A contains as row vectors the vectorized matrices 
    A_i for which <A_i,gamma>=b_i is supposed to hold.
    The positive real number r denotes the amount of regularization and can take values
    in the half-open interval [0, n_obs) with higher r demanding stronger regularization.
    
    Name                 Interpretation                             Type
    X                   Data matrix, each col is one vector-       Matrix [n,n_obs]
                        valued measurement.
    Lambda_p            Matrix prior for the psd coefficient       Matrix [n_exp,n_exp] 
                        tensor gamma 
    Psi                 Matrix containing info w.r.t the fun       Matrix [n,n_exp]
                        ction basis used for reconstruction.
                        Each col is one of the basis functions
                        measured by the measurement operator
    G                   Design matrix containing as entries        Matrix [n,n_mu]
                        the applications of the measurement 
    A                   Constraint matrix specifying the linear    Matrix [n_c, n_exp^2]
                        constraints A vec(gamma)=b
                        operator to the trend basis functions
    r                   Regularization parameter                   Number in [0, n_obs]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Tolerance for inversion procedures.        Small positive number
                        The larger the tolerance, the more 
                        regular S_psi is.
                        
    OUTPUTS
    The outputs consist in the coefficient tensor gamma to be used in the functional
    representation K= Sum_{ij} gamma_{ij}phi_i(.)phi_j(.) where the phi(.) are the
    basis functions stemming from the prior. For convenience, the chosen reconstruction
    C_gamma balancing fidelity to the empirical covariance matrix S_emp is given as
    well.The coefficient vector beta featuring in the representation mu=sum_i beta_i g_i
    together with the best guess G beta for the mean are provided. A logfile 
    containing the most important details of the optimization is provided in form 
    of the dictionary KI_logfile.
    
    Name                 Interpretation                             Type
    beta                Coefficient vector such that G beta        Vector [n_mu,1]
                        is the best guess for the mean mu
    mu_beta             Estimation of the mean function at         Vector [n,1]
                        the observed locations
    gamma               Coefficient tensor, is psd                 Matrix [n_exp,n_exp]
    S_emp               Empirical covariance matrix                Matrix [n,n]
    C_gamma             Reconstruction of the empirical            Matrix [n,n]
                        covariance matrix S_emp in terms of
                        the basis function evaluations in Psi
    KI_logfile          Dictionary containing details of  the      Dictionary
                        optimization
    
    """
    
    
    
    """
        1. Definitions and imports ------------------------------------------------
    """
    
    
    # i) Import packages
    
    import copy
    import numpy as np
    import numpy.linalg as lina
    import Support_funs as sf
    
    
    # ii) Extract dimensions from inputs
    
    n_exp=np.shape(Lambda_p)[0]
    n,n_obs=np.shape(X)
    
    tol=10**(-6)
    
    
    
    """
        2. Set up problem matrices -----------------------------------------------
    """
    
    
    # i) Problem matrices
    
    S_emp=(1/n_obs)*(X@X.T)
    Lambda_p_inv=lina.pinv(Lambda_p,hermitian=True)
    
    if type(A)==np.ndarray:
        b=A@np.reshape(Lambda_p,[n_exp**2,1])
        S_psi=sf.Get_S_psi(Psi, S_emp, A, b)   
    else: 
        S_psi=lina.pinv(Psi,rcond=tol)@S_emp@lina.pinv(Psi.T,rcond=tol)    
    
      
    
    
    """
        3. Perform iterations ----------------------------------------------------
    """
    
    
    # i) Initialize iteration
    
    gamma=copy.copy(Lambda_p)
    
    if type(G)==np.ndarray:
        n_mu=np.shape(G)[1]
        beta=np.zeros([n_mu,1])
    else:
        beta=0
        n_mu=0
        
    C_gamma=Psi@gamma@Psi.T
    norm_diff_max=1
    Backtracking_options=(0.5,0.5,max_iter)
    tol_obj=10**(-14)
    
    KI_logfile={"Convergence": "Yes" , "Nr_iter" : [], "Norm_difference" : [], "Objective_function":[] }
    
    
    # ii) Iteration
    
    k=1
    while norm_diff_max>10**(-6) and k < max_iter: 
        
        if type(G)==np.ndarray:
            
            # Step 1 Estimate beta
            B_beta_pinv=lina.pinv(G.T@lina.pinv(C_gamma,rcond=tol,hermitian=True)@G,rcond=tol,hermitian=True)
            beta_new=B_beta_pinv@G.T@lina.pinv(C_gamma,rcond=tol,hermitian=True)@X
            beta_new=np.reshape(np.mean(beta_new,1),[n_mu,1])
            
            Delta_beta=beta_new-beta
            beta=beta_new
            norm_diff_beta=np.max(np.abs(Delta_beta))
            
            
            # Step 2 Update S
            mu=G@beta
            mu_matrix=np.repeat(np.reshape(mu,[n,1]),n_obs,1)
            S_emp=(1/(n_obs))*((X-mu_matrix)@(X-mu_matrix).T)
            
            if type(A)==np.ndarray:
                S_psi=sf.Get_S_psi(Psi, S_emp, A, b)   
            else: 
                S_psi=lina.pinv(Psi,rcond=tol)@S_emp@lina.pinv(Psi.T,rcond=tol)
            
        else:
            norm_diff_beta=0
        
    
        # Step 3 Calculate Delta_gamma_total
        Delta_gamma_1=-(1/(n_obs+r))*((n_obs-r)*gamma-n_obs*S_psi+r*gamma@Lambda_p_inv@gamma)
    
        B_gg_pinv=(1/(n_obs+r))*(np.kron(gamma,gamma))
        
        if type(A)==np.ndarray:
            Mid_mat=lina.pinv(A@B_gg_pinv@A.T,rcond=tol,hermitian=True)
            Delta_gamma_2= -B_gg_pinv@(A.T@Mid_mat@A)@np.reshape(Delta_gamma_1,[n_exp**2,1])
            Delta_gamma_2=np.reshape(Delta_gamma_2,[n_exp,n_exp])
            Delta_gamma_total=Delta_gamma_1+Delta_gamma_2
        else:
            Delta_gamma_total=Delta_gamma_1
        
        Delta_gamma_total=0.5*(Delta_gamma_total+Delta_gamma_total.T)
        gamma_plus=lina.pinv(gamma,rcond=tol,hermitian=True)
        B_gg=(n_obs+r)*(np.kron(gamma_plus,gamma_plus))
        Newton_decrement=np.sqrt(np.reshape(Delta_gamma_total,[1,n_exp**2])@B_gg@np.reshape(Delta_gamma_total,[n_exp**2,1]))
        
         # Step 4 Backtracking linesearch
        def obj_fun(gamma):
            C_gamma=Psi@gamma@Psi.T
            obj_val_1=n_obs*(sf.Logpdet(C_gamma,tol_obj)+np.trace(S_emp@lina.pinv(C_gamma,rcond=tol_obj,hermitian=True)))
            obj_val_2=r*(-sf.Logpdet(gamma,tol_obj) + np.trace(Lambda_p_inv@gamma))
            Eigs=np.linalg.eigvals(gamma)
            bar_val=10**(20)*(np.sum((Eigs<-tol_obj).astype(int)))
    
            return obj_val_1+obj_val_2+bar_val
        
        t=sf.Backtracking_linesearch(obj_fun, gamma, Newton_decrement, Delta_gamma_total, Backtracking_options)
        
        
        # Step 5 Update gamma
        gamma=gamma+t*(Delta_gamma_total)
        gamma=0.5*(gamma+gamma.T)        
        C_gamma=Psi@gamma@Psi.T
        
        norm_diff_gamma=np.max(np.abs(t*Delta_gamma_total))
        norm_diff_max=np.max([norm_diff_gamma,norm_diff_beta])

        # Step 6 Update logfile
        k=k+1
        
        obj_val=obj_fun(gamma)
        KI_logfile["Norm_difference"]=np.append(KI_logfile["Norm_difference"],norm_diff_max)
        KI_logfile["Objective_function"]=np.append(KI_logfile["Objective_function"],obj_val)
        
        Message=('Executing damped Newton step with stepsize t = %f and objective '
                 'value L = %f' % (t,obj_val))
        print(Message)
        
        
        
    """
        4. Assemble solutions ----------------------------------------------------
    """
    
    
    # i) Assemble and return
    
    if k==max_iter:
        KI_logfile["Convergence"]="No"
        Message = ('The algorithm did not converge in the prespecified number of iterations. \n'
        'Handle results with care. Last update had norm %f' %norm_diff_max)
        print(Message)             
    KI_logfile["Nr_iter"]=k  
    
    C_gamma=Psi@gamma@Psi.T
    if type(G)==np.ndarray:
        mu_beta=G@beta
    else:
        mu_beta=0
    
    return beta, mu_beta, gamma, C_gamma, KI_logfile










def Kernel_inference_inhomogeneous(X_list, Lambda_p, Psi_list,r, G_list="none", A="none", max_iter=100, tol=10**(-2)):
      
    """
    The goal of this function is to provide an algorithm that solves the kernel 
    inference problem in presence of inhomogeneities, nonzeromeans and linear
    constraints. It is assumed that 
        i)      the data are gathered according to different rules (inhomogeneous)
        ii)     there is a trend hidden in the data (nonzeromean)
        iii)    constraints on the coefficients gamma are to be imposed (constrained))
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Perform iterations
        4. Assemble solutions
        
    INPUTS
    The inputs consist in a list X_list of data matrices, a prior Lambda_p on the 
    coefficient tensor gamma, and the list of matrices Psi containing the numerical 
    vectors yielded by applying the measurement operators to the sequence of 
    eigenfunctions from the Mercer decomposition of the prior.
    The positive real number r denotes the amount of regularization and can take 
    values in the half-open interval [0, n_obs] with higher r demanding stronger 
    regularization.
    
    Name                 Interpretation                             Type
    X_list              List of data matrices, in each matrix,      List [n^S_obs]
                        each col is one vector-valued               Matrix [n_i,n^S_i]
                        measurement.
    Lambda_p            Matrix prior for the psd coefficient        Matrix [n_exp,n_exp] 
                        tensor gamma 
    Psi_list            List of matrices, each matrix contains      List [n_S_obs]  
                        info w.r.t the function basis used for      Matrix [n_i,n_exp]
                        reconstruction.
                        Each col is one of the basis functions
                        measured by the measurement operator
                        operator to the trend basis functions
    G_list              List of design matrices containing as      List [n_S_obs] 
                        entries the applications of the            Matrix [n_i,n_mu]
                        measurementoperators to the trend basis
                        functions.
    A                   Constraint matrix specifying the linear    Matrix [n_c, n_exp^2]
                        constraints A vec(gamma)=b
    r                   Regularization parameter                   Number in [0, n_obs]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Tolerance for inversion procedures.        Small positive number
                        The larger the tolerance, the more 
                        regular S_psi is.
    
                        
    OUTPUTS
    The outputs consist in the coefficient tensor gamma to be used in the functional
    representation K= Sum_{ij} gamma_{ij}phi_i(.)phi_j(.) where the phi(.) are the
    basis functions stemming from the prior. For convenience, the chosen reconstruction
    C_gamma balancing fidelity to the empirical covariance matrix S_emp is given as
    well.The coefficient vector beta featuring in the representation mu=sum_i beta_i g_i
    together with the best guess G beta for the mean are provided. A logfile 
    containing the most important details of the optimization is provided in form 
    of the dictionary KI_logfile.
    
    Name                 Interpretation                             Type
    beta                Coefficient vector such that G beta        Vector [n_mu,1]
                        is the best guess for the mean mu
    mu_beta             Estimation of the mean function at         Vector [n,1]
                        the observed locations
    gamma               Coefficient tensor, is psd                 Matrix [n_exp,n_exp]
    gamma_tilde         Full coefficient matrix consisting in      Matrix [n_total,n_total]
                        block diagonal composition of gamma 
                        and all the eta's.'
    C_gamma_tilde       Reconstruction of the empirical            Matrix [sum n_i, sum n_i]
                        covariance matrices S_emp in terms of
                        the basis function evaluations in Psi                      
    KI_logfile          Dictionary containing details of  the      Dictionary
                        optimization
    
    """
    
    
    
    """
        1. Definitions and imports ------------------------------------------------
    """
    
    
    # i) Import packages
    
    import copy
    import numpy as np
    import numpy.linalg as lina
    import scipy.linalg as spla
    import Support_funs as sf
    
    
    # ii) Extract dimensions from inputs
    
    n_S_obs=len(X_list)
    n_exp=np.shape(Lambda_p)[0]
    
    n_i=[]
    n_S_i=[]
    for X in X_list:
        n_i.append(np.shape(X)[0])
        n_S_i.append(np.shape(X)[1])
        
    tol=10**(-6)
    
    
    
    """
        2. Set up problem matrices -----------------------------------------------
    """
    
    
    # i) Problem matrices
    
    S_emp_list=[]
    S_psi_list=[]
    S_phi_list=[]
    
    Phi_list=[]
    Phi_Psi_list=[]
    
    for k in range(n_S_obs):
        S_emp_list.append((1/n_S_i[k])*(X_list[k]@X_list[k].T))
        [Phi_temp,del_temp_1,del_temp_2]=lina.svd(S_emp_list[k])
        Phi_list.append(Phi_temp)
        Phi_Psi_list.append(Phi_temp.T@Psi_list[k])
        
        S_phi_list.append(Phi_temp.T@S_emp_list[k]@Phi_temp)
        S_psi_list.append(lina.pinv(Psi_list[k],rcond=tol)@S_emp_list[k]@lina.pinv(Psi_list[k].T,rcond=tol))
                          
    Lambda_p_inv=lina.pinv(Lambda_p,hermitian=True)
      
    
    # ii) Construct affine constraints:  Block diagonal constraints F_i gamma-eta=0
     
    A_dg_list=[]
    n_i_temp=np.array([0])
    for k in range(n_S_obs):
        n_i_temp=np.hstack((n_i_temp,n_i[k]))
    
    n_i_cumsum=np.cumsum(n_i_temp)
    n_i_sum=np.sum(n_i_temp)
    A_dg=np.zeros([0,(n_exp+n_i_sum)**2])
    
    for k in range(n_S_obs):
        
        n_first_zeros=n_i_cumsum[k]
        n_second_zeros=n_i_sum-n_i_cumsum[k+1]
        
        B_1=np.hstack((Phi_Psi_list[k],np.zeros([n_i[k],n_first_zeros]),np.eye(n_i[k]),np.zeros([n_i[k],n_second_zeros])))
        B_2=np.hstack((Phi_Psi_list[k],np.zeros([n_i[k],n_first_zeros]),-np.eye(n_i[k]),np.zeros([n_i[k],n_second_zeros])))
        A_dg_k=np.kron(B_1,B_2)
        
        A_dg_list.append(A_dg_k)
        A_dg=np.vstack((A_dg,A_dg_k))
    
    
    # iii) Construct affine constraints: Off diagonal constraints Offdiag(gamma_tilde)=0
    
    n_total=n_exp+n_i_sum
    Indicator_mat=np.ones([n_exp,n_exp])
    
    for k in range(n_S_obs):
        Indicator_mat=spla.block_diag(Indicator_mat,np.ones([n_i[k],n_i[k]]))    
    Off_diag_mat=np.ones([n_total,n_total])-Indicator_mat
    
    Index_tuple=np.nonzero(Off_diag_mat)
    nd_index=Index_tuple[0]
    lin_ind=np.ravel_multi_index(Index_tuple,[n_total,n_total])
    n_nondiag=len(nd_index)
    
    A_nd=np.zeros([0,n_total**2])
    for k in range(n_nondiag):
        index_temp=np.zeros([1,n_total**2])
        index_temp[0,lin_ind[k]]=1
        A_nd=np.vstack((A_nd,index_temp))
    
    
    # iv) Construct affine constraints: Original constraints encoded by A
    
    if type(A)==np.ndarray:
        n_c=A.shape[0]
        A_tilde=np.zeros([n_c,n_total**2])
        
        for k in range(n_c):
            a_i_mat=np.reshape(A[k,:],[n_exp,n_exp])
            a_i_tilde_mat=np.zeros([n_total,n_total])
            a_i_tilde_mat[:n_exp,:n_exp]=a_i_mat
            A_tilde[k,:]=np.reshape(a_i_tilde_mat,[1,n_total**2])
            
        A_c=np.vstack((A_tilde, A_dg, A_nd))
    else:
        A_c=np.vstack((A_dg, A_nd))
    
    
    
    """
        3. Perform iterations ----------------------------------------------------
    """
    
    
    # i) Initialize iteration
    
    # Step 1 Set up matrices
    gamma=copy.copy(Lambda_p)
    gamma_tilde_list=[gamma]
    eta_list=[]
    C_gamma_list=[]
    gamma_tilde=gamma
    
    if type(G_list[0])==np.ndarray:
        n_mu=G_list[0].shape[1]
        beta=np.zeros([n_mu,1])
    else:
        n_mu=0
        beta=0
    
    # Step 2 Initialize options            
    Backtracking_options=(0.5,0.5,max_iter)
    tol_obj=10**(-8)
    
    KI_logfile={"Convergence": "Yes" , "Nr_iter" : [], "Norm_difference" : [], "Objective_function":[] }
    
    # Step 3 set up lists
    for k in range(n_S_obs):
        gamma_tilde_list.append(Phi_Psi_list[k]@gamma@Phi_Psi_list[k].T)
        eta_list.append(gamma_tilde_list[k+1])
        gamma_tilde=spla.block_diag(gamma_tilde,gamma_tilde_list[k+1]) 
        C_gamma_list.append(Phi_list[k]@eta_list[k]@Phi_list[k].T)


    # ii) Iteration
    
    norm_diff_max=1
    step_counter=1
    while norm_diff_max>10**(-6) and step_counter < max_iter: 
        
        if type(G_list[0])==np.ndarray:         # When nontrivial mean functions present
            
            # Step 1 Update beta
            B_beta=np.zeros([n_mu,n_mu])
            grad_beta=np.zeros([n_mu,1])
            
            for k in range(n_S_obs):
                C_gamma_temp_pinv=lina.pinv(C_gamma_list[k],rcond=tol, hermitian=True)
                B_beta=B_beta+n_S_i[k]*((G_list[k].T)@(C_gamma_temp_pinv)@(G_list[k]))
                grad_beta=grad_beta+np.reshape((n_S_i[k])*(G_list[k].T)@(C_gamma_temp_pinv)@np.mean(X_list[k],axis=1),[n_mu,1])
                
            B_beta_pinv=lina.pinv(B_beta,rcond=tol,hermitian=True) 
            beta_new=B_beta_pinv@grad_beta
            Delta_beta= -beta+beta_new
            beta=beta_new
            norm_diff_beta=np.max(np.abs(Delta_beta))
            
            
            # Step 2 Update S_emp_list
            S_emp_list=[]
            S_phi_list=[]
            for k in range(n_S_obs):
                mu_temp=G_list[k]@beta
                mu_matrix_temp=np.repeat(np.reshape(mu_temp,[n_i[k],1]),n_S_i[k],1)
                S_emp_list.append((1/(n_S_i[k]))*((X_list[k]-mu_matrix_temp)@(X_list[k]-mu_matrix_temp).T))
                S_phi_list.append(lina.pinv(Phi_list[k],rcond=tol)@S_emp_list[k]@lina.pinv(Phi_list[k].T,rcond=tol))      
            
            # Step 3 Update objective function
            def obj_fun(gamma_var,eta_list_var):
                obj_val_1=0
                for k in range(n_S_obs):
                    temp_1=n_S_i[k]*(sf.Logpdet(eta_list_var[k],tol_obj)+np.trace(S_phi_list[k]@lina.pinv(eta_list_var[k],rcond=tol_obj,hermitian=True)))
                    obj_val_1=obj_val_1+temp_1
                obj_val_2=r*(-sf.Logpdet(gamma_var,tol_obj) + np.trace(Lambda_p_inv@gamma_var))
            
                Eigs=np.linalg.eigvalsh(gamma_var)
                bar_val=10**(20)*(np.sum((Eigs<-tol_obj).astype(int)))
                for k in range(n_S_obs):
                    Eigs=np.linalg.eigvalsh(eta_list_var[k])
                    temp_2=10**(10)*(np.sum((Eigs<-tol_obj).astype(int)))
                    bar_val=bar_val+temp_2
        
                return obj_val_1+obj_val_2+bar_val
            
            
        else:                                    # When nontrivial mean functions absent
            # Step 1 Set up S_emp_list
            S_emp_list=[]
            S_phi_list=[]
            for k in range(n_S_obs):
                S_emp_list.append((1/(n_S_i[k]))*((X_list[k])@(X_list[k]).T))
                S_phi_list.append(lina.pinv(Phi_list[k],rcond=tol)@S_emp_list[k]@lina.pinv(Phi_list[k].T,rcond=tol))      
            
            # Step 2 Set up objective function
            def obj_fun(gamma_var,eta_list_var):
                obj_val_1=0
                for k in range(n_S_obs):
                    temp_1=n_S_i[k]*(sf.Logpdet(eta_list_var[k],tol_obj)+np.trace(S_phi_list[k]@lina.pinv(eta_list_var[k],rcond=tol_obj,hermitian=True)))
                    obj_val_1=obj_val_1+temp_1
                obj_val_2=r*(-sf.Logpdet(gamma_var,tol_obj) + np.trace(Lambda_p_inv@gamma_var))
            
                Eigs=np.linalg.eigvalsh(gamma_var)
                bar_val=10**(20)*(np.sum((Eigs<-tol_obj).astype(int)))
                for k in range(n_S_obs):
                    Eigs=np.linalg.eigvalsh(eta_list_var[k])
                    temp_2=10**(10)*(np.sum((Eigs<-tol_obj).astype(int)))
                    bar_val=bar_val+temp_2
        
                return obj_val_1+obj_val_2+bar_val
            
            norm_diff_beta=0
        
        
        # Step 4 Update Delta_gamma_tilde_1
        grad_gamma_tilde=r*(lina.pinv(gamma,rcond=tol,hermitian=True)-Lambda_p_inv)
        for k in range(n_S_obs):
            eta_pinv=lina.pinv(eta_list[k],rcond=tol,hermitian=True)
            grad_gamma_tilde=spla.block_diag(grad_gamma_tilde,n_S_i[k]*(eta_pinv@S_phi_list[k]@eta_pinv-eta_pinv))

        Q=(1/np.sqrt(r))*(gamma)
        Q_pinv=np.sqrt(r)*(lina.pinv(gamma,rcond=tol,hermitian=True))
        
        for k in range(n_S_obs):
            Q=spla.block_diag(Q,(1/np.sqrt(n_S_i[k]))*eta_list[k])
            Q_pinv=spla.block_diag(Q_pinv,(np.sqrt(n_S_i[k]))*(lina.pinv(eta_list[k],rcond=tol,hermitian=True)))
        
        Delta_gamma_tilde_1=Q@grad_gamma_tilde@Q
        Delta_gamma_tilde_1=0.5*(Delta_gamma_tilde_1+Delta_gamma_tilde_1.T)
   
         
        # Step 5 Update Delta_gamma_tilde_2              
        B_gg_pinv=np.kron(Q,Q)
        B_gg=np.kron(Q_pinv,Q_pinv)
            

        Mid_mat=lina.pinv(A_c@B_gg_pinv@A_c.T,rcond=tol,hermitian=True)
        Delta_gamma_tilde_2= -B_gg_pinv@(A_c.T@Mid_mat@A_c)@np.reshape(Delta_gamma_tilde_1,[n_total**2,1])
        Delta_gamma_tilde_2=np.reshape(Delta_gamma_tilde_2,[n_total,n_total])
        Delta_gamma_tilde_total=Delta_gamma_tilde_1+Delta_gamma_tilde_2
        Delta_gamma_tilde_total=0.5*(Delta_gamma_tilde_total+Delta_gamma_tilde_total.T)

        
        # Step 6 Backtracking linesearch
        Delta_gamma_total=Delta_gamma_tilde_total[:n_exp,:n_exp]
        Delta_eta_list=[]
        for k in range(n_S_obs):
            lower_index=n_exp+n_i_cumsum[k]
            upper_index=n_exp+n_i_cumsum[k+1]
            Delta_eta_list.append(Delta_gamma_tilde_total[lower_index:upper_index,lower_index:upper_index])
            
        Newton_decrement=np.sqrt(np.reshape(Delta_gamma_tilde_total,[1,n_total**2])@B_gg@np.reshape(Delta_gamma_tilde_total,[n_total**2,1]))
        t=sf.Backtracking_linesearch_inhomogeneous(obj_fun, gamma, eta_list,  Newton_decrement, Delta_gamma_total, Delta_eta_list, Backtracking_options)
        
        gamma_tilde=gamma_tilde+t*(Delta_gamma_tilde_total)
        gamma_tilde=0.5*(gamma_tilde+gamma_tilde.T)
        
        
        # Step 7 Update coefficients
        gamma=gamma_tilde[:n_exp,:n_exp]
        
        for k in range(n_S_obs):
            lower_index=n_exp+n_i_cumsum[k]
            upper_index=n_exp+n_i_cumsum[k+1]
            eta_list[k]=gamma_tilde[lower_index:upper_index,lower_index:upper_index]
                                                                              
        
        # Step 8 Update logfile
        norm_diff_gamma=np.max(np.abs(t*Delta_gamma_tilde_total))
        norm_diff_max=np.max([norm_diff_gamma, norm_diff_beta])      
        step_counter=step_counter+1
        
        obj_val=obj_fun(gamma, eta_list)
        KI_logfile["Norm_difference"]=np.append(KI_logfile["Norm_difference"],norm_diff_max)
        KI_logfile["Objective_function"]=np.append(KI_logfile["Objective_function"],obj_val)

        Message=('Executing damped Newton step with stepsize t = %f and objective '
                 'value L = %f' % (t,obj_val))
        print(Message)
        
        
        
    """
        4. Assemble solutions ----------------------------------------------------
    """
 
    
    # i) Assemble and return
    
    gamma=gamma_tilde[:n_exp,:n_exp]
    C_gamma_tilde=np.zeros([0,0])
    mu_beta=np.zeros([0,1])
    
    for k in range(n_S_obs):
        eta_temp=gamma_tilde[n_exp+n_i_cumsum[k]:n_exp+n_i_cumsum[k+1],n_exp+n_i_cumsum[k]:n_exp+n_i_cumsum[k+1]]    
        C_gamma_tilde=spla.block_diag(C_gamma_tilde,(Phi_list[k])@eta_temp@(Phi_list[k]).T)
        
        if type(G_list[0])==np.ndarray:
            mu_beta=np.vstack((mu_beta,G_list[k]@beta))
        
        
    # ii) Update logfile
    
    if step_counter==max_iter:
        KI_logfile["Convergence"]="No"
        Message = ('The algorithm did not converge in the prespecified number of iterations. \n'
         'Handle results with care. Last update had norm %f' %norm_diff_max)
        print(Message)       
    KI_logfile["Nr_iter"]=step_counter  
        
        
    return  beta, mu_beta, gamma_tilde, C_gamma_tilde, gamma, KI_logfile

