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
       
        min_eig = np.inf
        max_eig = -np.inf

        for name,data in self.data_groups.items():

            eigvals,eigvecs = run_PCA(data)
            k_largest = eigvecs[:,-self.n_components:]
            group_loss_k = loss(data,k_largest)
            self.ideal_loss[name] = group_loss_k

            min_eig = min(min_eig,eigvals[0])
            max_eig = max(max_eig,eigvals[-1])

        gap = max_eig - min_eig
        self.alpha = gap + 0.001

    def square(self,x):
        return 0.5*jnp.power(x,2)

    def exp(self,x):
        return jnp.exp(x)

    def compute_gradients(self):

        # needed to turn into a strongly convex problem
        regularization = 2*self.alpha*self.U

        U_grad = -2*self.data.T @ self.data @ self.U + regularization
        grads = [U_grad]

        subgroup_loss = []

        for name,data in self.data_groups.items():
            group_loss = loss(data,self.U) - self.ideal_loss[name]
            subgroup_loss.append(group_loss)

        if self.pairwise:
            # k choose 2 pairwise differences
            for i,j  in combinations(range(len(self.data_groups)),2):
                delta_ij = subgroup_loss[i] - subgroup_loss[j]        
                pairwise_grad = self.penalty_grad(delta_ij) + regularization 
                print("G_{}{} : {}".format(i,j,pairwise_grad.shape))
                grads.append(pairwise_grad)
        else:
            # directly compute on subgroups
            for i in range(subgroup_loss):
                grad = self.penalty_grad(subgroup_loss[i]) + regularization
                grads.append(grad)

        return grads
        
    def run(self,max_iter):

        self.U = np.random.randn(self.data.shape[1],self.n_components)

        for t in range(max_iter):
            gradients = self.compute_gradients() 
            direction = self.select_direction(gradients)
            
            # zero update direction indicates no possible Pareto-efficient improvement
            if np.all(direction == 0.0):
                return self.U
            
            # decreasing learning rate
            learning_rate = 1.0 / np.sqrt(t)
            self.U = self.projected_gradient(self.U,direction,learning_rate)

        return self.U

    def select_direction(self,gradients):
        # solve quadratic programming problem for dual function

        n = len(gradients)
        coeffs = cp.Variable(n)
        objective = cp.norm(coeffs.T @ gradients,p="fro")
        
        ones = np.ones(n)
        zeros = np.zeros(n)
        basis = np.eye(n)
        
        # constraints enforce probability simplex
        sum_to_one = ones.T @ coeffs == 1
        non_negative =  -basis @ coeffs <= zeros
        
        prob = cp.Problem(cp.Minimize(objective),[sum_to_one,non_negative])
        prob.solve()

        lambda_hat = prob.value.tolist()
        direction = np.zeros_like(self.U)

        for i,l in enumerate(lambda_hat):
            direction += l*gradients[i]

        return direction

    def projected_gradient(self,current,direction,learning_rate):
        
        # use singular value decomposition to find orthogonal projection of update
        update = current + learning_rate*direction

        u,s,vh = np.linalg.svd(update)
        return vh @ vh.T

def eigenstrat(data : np.ndarray,n_components):

    # demean and normalize, run PCA
    processed = preprocess(data)
    eig_vals,princ_comps = run_PCA(processed,n_components=n_components)
   
    #  multilinear  regression on principal components
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
    
    pca = FairPCA(full_data,data_groups,100)
    U = pca.run(10000)

    #stratified = eigenstrat(snp_test,n_components=10)
    
