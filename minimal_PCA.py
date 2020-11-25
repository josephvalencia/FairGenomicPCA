import sys
import gzip
import numpy as np
from scipy.linalg import eigh
from itertools import combinations
from collections import defaultdict

import jax.numpy as jnp
from jax import grad
import cvxpy as cp

def run_PCA(data : np.ndarray,n_components=None):
   
    corr = data.T @ data

    if n_components is not None:
        n,m = corr.shape
        subset = [m-n_components,m-1] #n_components largest
        return eigh(corr,subset_by_index=subset)
    else:
        return eigh(corr)

def loss(data : np.ndarray, projector):

    reconstruction = data  @ projector @ projector.T
    return np.linalg.norm(data - reconstruction)**2

class FairPCA:

    def __init__(self,data,groups,n_components,phi = "square",pairwise=True):
        
        self.data = data
        self.data_groups = groups
        self.n_components = n_components
        self.pairwise = pairwise

        if phi == "square":
            self.penalty = self.square
        elif phi == "exp":
            self.penalty = self.exp
        else:
            raise ValueError("penalty function must be \'square\' or \'exp\'")
   
        self.penalty_grad = grad(self.penalty)

        self.optimal_group_projections()

    def optimal_group_projections(self):

        self.ideal_loss = defaultdict(float)
       
        max_min_eig = -np.inf
        min_max_eig = np.inf

        for name,data in self.data_groups.items():

            eigvals,eigvecs = run_PCA(data)
            k_largest = eigvecs[:,-self.n_components:]
            group_loss_k = loss(data,k_largest)
            self.ideal_loss[name] = group_loss_k

            max_min_eig = max(max_min_eig,eigvals[0])
            min_max_eig = min(min_max_eig,eigvals[-1])

        gap = max_min_eig - min_max_eig
        self.alpha = gap + 0.001

    def square(self,x):
        return 0.5*jnp.power(x,2)

    def exp(self,x):
        return jnp.exp(-x)

    def compute_gradients(self):

        # needed to turn into a strongly convex problem
        regularization = 2*self.alpha*self.U
        U_grad = -2*self.data.T @ self.data @ self.U + regularization
        grads = [U_grad]

        subgroup_loss = []
        subgroup_grad = []

        for name,data_group in self.data_groups.items():
            group_loss = loss(data_group,self.U) - self.ideal_loss[name]
            subgroup_loss.append(group_loss)
            group_grad = -2*data_group.T @ data_group @ self.U
            subgroup_grad.append(group_grad)

        if self.pairwise:
            # k choose 2 pairwise differences
            for i,j  in combinations(range(len(self.data_groups)),2):
                delta_ij = subgroup_loss[i] - subgroup_loss[j]        
                print("Delta_{}{} = {}".format(i,j,delta_ij))
                chain = subgroup_grad[i] + subgroup_grad[j]
                print("chain",chain.shape)
                pairwise_grad = self.penalty_grad(delta_ij) @ chain + regularization 
                pairwise_grad = pairwise_grad / (np.linalg.norm(pairwise_grad) + 1e-32)
                grads.append(pairwise_grad)
        else:
            # directly compute on subgroups
            for i in range(len(subgroup_loss)):
                grad = self.penalty_grad(subgroup_loss[i]) @ subgroup_grad[i] + regularization
                grad = grad / (np.linalg.norm(grad)+1e-32)
                grads.append(grad)

        return grads
        
    def run(self,max_iter):

        self.U = np.random.randn(self.data.shape[1],self.n_components)

        for t in range(1,max_iter):
            print("t",t)
            gradients = self.compute_gradients() 
            direction = self.select_direction(gradients)
            
            # zero update direction indicates no possible Pareto-efficient improvement
            if np.all(direction == 0.0):
                return self.U
            
            # decreasing learning rate
            learning_rate = 1.0 / np.sqrt(t)
            self.U = self.orthogonal_projection(self.U,direction,learning_rate)

        return self.U

    def select_direction(self,gradients):
        # solve quadratic programming problem for dual function

        n = len(gradients)
        x = cp.Variable(n)

        # vectorize gradients for each objective and pack into a single matrix
        #G = np.stack(gradients,axis=2)        
        #G = G.reshape(-1,n)
        
        G = []

        for g in gradients:
            dim,n_components = g.shape
            vec = g.reshape(-1)
            G.append(vec)
        
        G = np.stack(G,axis=1)
        P = G.T @ G
        eigvals,eigvecs = np.linalg.eigh(P)
        print("eigvals",eigvals)
        
        objective = (1/2)*cp.quad_form(x,P)
        
        ones = np.ones(n)
        zeros = np.zeros(n)
        basis = np.eye(n)
        
        # constraints enforce probability simplex
        sum_to_one = ones.T @ x == 1
        non_negative = -basis @ x <= zeros
        
        prob = cp.Problem(cp.Minimize(objective),[non_negative,sum_to_one])
        prob.solve()

        lambda_hat = x.value
        direction = G @ lambda_hat
        direction = direction.reshape(dim,n_components)

        return -direction

    def orthogonal_projection(self,current,direction,learning_rate):
        
        # use singular value decomposition to find orthogonal projection of update
        update = current + learning_rate*direction
        
        # orthogonal Procrustes problem
        u,s,vh = np.linalg.svd(update,full_matrices=False)
        orth_proj = u @ vh
        
        return orth_proj

def eigenstrat(data : np.ndarray,n_components):

    # demean and normalize, run PCA
    processed = preprocess(data)
    eig_vals,princ_comps = run_PCA(processed,n_components=n_components)
   
    #  multilinear regression on principal components
    pc_coeff = np.matmul(data,princ_comps) / np.linalg.norm(princ_comps,axis=0,ord=2)**2 
    regression = np.matmul(pc_coeff,princ_comps.T)

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

if __name__ == "__main__":

    #vcf_file = sys.argv[1]
    #parse_VCF(vcf_file)

    data = []
    
    for k in range(5):
        snp_test = 1.0*np.random.randint(low=0,high=3,size=(1000,1000))
        data.append(snp_test)

    full_data = np.concatenate(data,axis=0)
    data_groups = { i : group for i,group in enumerate(data) }
    
    pca = FairPCA(full_data,data_groups,100,pairwise=False)
    U = pca.run(10000)

    #stratified = eigenstrat(snp_test,n_components=10)
    
