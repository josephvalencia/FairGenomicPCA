import numpy as np
from PCA import run_PCA , FairPCA , frank_wolfe_NSW 

def eigenstrat(data : np.ndarray,pca_alg = "vanilla",n_components):

    # demean and normalize, run PCA
    processed = preprocess(data)

    if pca_alg == "vanilla":
        eig_vals,princ_comps = run_PCA(processed,n_components=n_components)
    elif pca_alg == "pareto":
        fair = FairPCA()
    elif pca_alg == "nash":
        X = frank_wolfe_NSW()

    #  multilinear regression on principal components
    pc_coeff = data @ princ_comps / np.linalg.norm(princ_comps,axis=0,ord=2)**2 
    regression = pc_coeff @ princ_comps.T

    # subtract fitted values to remove population effect
    corrected = data-regression
    return corrected

def preprocess(data : np.ndarray):

    M,N = data.shape
    
    # subtract row means
    mu =  np.mean(data,axis=1)
    mu = mu.reshape(-1,1)
    data -= mu
    
    # normalize row by posterior
    p_i = (1+mu*N)/(2+2*N)
    p_i_comp = np.ones_like(p_i) - p_i
    norm_term = np.sqrt(np.dot(p_i.T,p_i_comp))
    data /= norm_term
    return data

