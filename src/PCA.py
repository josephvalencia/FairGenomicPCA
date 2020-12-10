import sys
import gzip
import numpy as np
import time
from scipy.linalg import eigh
from itertools import combinations
from collections import defaultdict
from sklearn import preprocessing
import jax.numpy as jnp
from jax import grad
import cvxpy as cp

#from src.load_data import parse_VCF

def run_PCA(data : np.ndarray,n_components=None):
   
    corr = data.T @ data

    if n_components is not None:
        n,m = corr.shape
        subset = [m-n_components,m-1] # n_components largest
        return eigh(corr,subset_by_index=subset)
    else:
        return eigh(corr)

def loss(data : np.ndarray, projector):

    reconstruction = data  @ projector @ projector.T
    return np.linalg.norm(data - reconstruction,ord='fro')**2

def frank_wolfe_NSW(all_data,data_groups,d,learning_rate,epsilon):

    # optimize Nash social welfare objective using Frank-White Algorithm
    B_list = []

    for name,data in data_groups.items():
        m,n = data.shape
        B = data.T @ data
        B_list.append(B)
        print("group {} cov matrix {}".format(name,B.shape))

    X = d/n * np.eye(N=n,M=n)
    dual_gap = np.inf
    alpha = 0.5
    print("A")
    welfare = nash_social_welfare(B_list,X,log=True,alpha=alpha)
    print("B")
    eigvals,eigvecs = np.linalg.eigh(np.cov(all_data.T))
    print("C")
    k_largest = eigvecs[:,-d:]
    ideal_loss = loss(all_data,k_largest)
    curr_loss = loss(all_data,X)
    print("D")
    
    print("Nash social welfare", welfare)
    print("Ideal Reconstruction loss",ideal_loss) 
    print("initial Reconstruction loss",curr_loss)

    while dual_gap > epsilon:
        
        gradient = np.zeros(shape=(n,n))
        
        for group in B_list:
            group_grad = 1 / (np.trace(group @ X) + alpha * np.linalg.norm(group,ord='fro'))
            gradient += group_grad * group

        # mode 1
        w,v = np.linalg.eigh(gradient)
        idx = np.argsort(w)
        w = w[idx]
        v = v[:,idx]
        v_top = v[:,-d:]
        
        '''
        s = time.time()
        # mode 2 
        subset = [n-d,n-1] # n_components largest
        w,v = eigh(gradient,subset_by_index=subset)
        e = time.time()
        b_times.append(e-s)
        '''
        oracle = v_top @ v_top.T
        dual_gap = np.trace(gradient.T @ (oracle-X))
        X = (1-learning_rate) * X + learning_rate*oracle
        welfare = nash_social_welfare(B_list,X,alpha=alpha,log=True)
        
        print("dual_gap = {}, NSW = {}, tr(X) = {}".format(dual_gap,welfare,np.trace(X)))
        reconstruction = np.linalg.norm(all_data - all_data @ X)**2
        print("reconstruction error = {} , ideal error = {}".format(reconstruction,ideal_loss))
    
    return X

def nash_social_welfare(B_list,X,log=True,alpha = 0.0):

    if log:
        welfare = 0
        for group in B_list:
            welfare +=  np.log(np.trace(group @ X))  #+ alpha * np.trace(group)**2)
    else:
        welfare = 1
        for group in B_list:
            welfare *= np.trace(group @ X) + alpha*np.trace(group)**2 

    return welfare

def factors(X,d):

    # return matrix P size(n,d)  such that P.T @ P == I_d and X = P @ P.T
    
    w,v = np.linalg.eigh(X)
    idx = np.argsort(-w)
    w = w[idx]
    v = v[:,idx]
    w = w[:d]
    v_top = v[:,:d]

    lambda_mat = np.diag(w)

    P = v_top @ lambda_mat
    return P

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
        elif phi == "abs":
            self.penalty = self.abs
        else:
            raise ValueError("penalty function must be \'square\' or \'exp\'")
   
        self.penalty_grad = grad(self.penalty)
        self.loss_progress = [np.inf]*(len(self.data_groups)+1)
        #self.optimal_group_projections()
        self.alpha = 10.0

    def optimal_group_projections(self):

        self.ideal_loss = defaultdict(float)
       
        max_min_eig = -np.inf
        min_max_eig = np.inf

        for name,data in self.data_groups.items():

            #eigvals,eigvecs = run_PCA(data)
            eigvals,eigvecs = np.linalg.eigh(all_data.T @ all_data)
            k_largest = eigvecs[:,-self.n_components:]
            group_loss_k = loss(data,k_largest)
            self.ideal_loss[name] = group_loss_k

            max_min_eig = max(max_min_eig,eigvals[0])
            min_max_eig = min(min_max_eig,eigvals[-1])
            print("max eig {}, min eig {} , ideal loss {}".format(eigvals[-1],eigvals[0],group_loss_k))

        gap = min_max_eig - max_min_eig
       
        '''
        if gap < 0:
            self.alpha = 0.1
        else:
            self.alpha = gap + 0.1
        '''
        
        self.alpha =  0.5
        print("min_max_eig {} , max_min_eig {}, gap {}, alpha {}".format(min_max_eig,max_min_eig,gap,self.alpha))

    def square(self,x):
        return 0.5*jnp.power(x,2)

    def exp(self,x):
        return jnp.exp(-x)

    def abs(self,x):
        return jnp.abs(x)

    def compute_gradients(self):

        # needed to turn into a strongly convex problem
        reg_grad = 2*self.alpha*self.U
        #reg_grad = reg_grad / np.linalg.norm(reg_grad)

        U_grad = -2*self.data.T @ self.data @ self.U
        #U_grad = U_grad / np.linalg.norm(U_grad)
       
        #U_grad = U_grad + reg_grad 
        #U_grad = U_grad / np.linalg.norm(U_grad) +reg_grad
        grads = [U_grad]

        #print("Overall loss {}".format(loss(self.data,self.U)))

        '''        
        subgroup_loss = []
        subgroup_grad = []

        for name,data_group in self.data_groups.items():
            group_loss = loss(data_group,self.U)
            group_ideal = self.ideal_loss[name]
            subgroup_loss.append(group_loss-group_ideal)
            group_grad = -2*data_group.T @ data_group @ self.U
            #print(name,group_grad)
            subgroup_grad.append(group_grad)

        if self.pairwise:
            # k choose 2 pairwise differences
            for i,j  in combinations(range(len(self.data_groups)),2):
                delta_ij = subgroup_loss[i] - subgroup_loss[j]        
                #print("delta_{}{} = {}".format(i,j,delta_ij))
                chain = subgroup_grad[i] - subgroup_grad[j]
                #print("chain",chain)
                pairwise_grad =  self.penalty_grad(delta_ij) * chain
                
                pairwise_grad = pairwise_grad / np.linalg.norm(pairwise_grad)
                pairwise_grad = pairwise_grad + reg_grad

                grads.append(pairwise_grad)
        else:
            # directly compute on subgroups
            for i in range(len(subgroup_loss)):
                grad = self.penalty_grad(subgroup_loss[i]) * subgroup_grad[i] + reg_grad 
                grad = grad / np.linalg.norm(grad)  
                grads.append(grad)
        '''
        return grads
   
    def track_progress(self):

        disparities  = []        
        overall_loss = loss(self.data,self.U)
        print("overall loss {}".format(overall_loss))

        for name,data_group in self.data_groups.items():
            group_loss = loss(data_group,self.U) - self.ideal_loss[name]
            disparities.append(group_loss)

        if self.pairwise:
            for i,j  in combinations(range(len(self.data_groups)),2):
                delta_ij = disparities[i] - disparities[j]
                print("delta_{}{} = {}".format(i,j,delta_ij))
        else:
            for i in range(len(self.data_groups)):
                print("loss group {} = {}".format(i,disparities[i]))

        '''
        for i in range(len(new_scores)):
            if new_scores[i] > self.loss_progress[i]:
                #print("Error, something got worse!")
                #print("old_scores : {}".format(self.loss_progress))
                #print("new scores : {}".format(new_scores))
                self.loss_progress = new_scores
                return False

        self.loss_progress = new_scores
        
        return True
        '''

    def run(self,max_iter):

        self.U = np.random.randn(self.data.shape[1],self.n_components)
        total = self.data.shape[1] * self.n_components
        
        for t in range(1,max_iter):
            
            gradients = self.compute_gradients() 
            
            #direction = self.select_direction(gradients)
            direction = -gradients[0]

            if t < 10 or t % 10 == 0:
                print("iteration {}".format(t))
                improving = self.track_progress()
                #print("norm direction =",np.linalg.norm(direction))
                #print("direction",direction)
                #print("U",self.U)

            U_norm = np.linalg.norm(self.U)
            if U_norm == 0.0:
                print(direction)
                quit()
            
            # zero update direction indicates no possible Pareto-efficient improvement
            small = direction <= 1e-5

            # decreasing learning rate
            learning_rate = 1. / np.sqrt(t)
       
            cache = self.U

            update = self.U + learning_rate*direction
            self.U = self.orthogonal_projection(update)
            #print("delta U",np.linalg.norm(self.U - cache))

        return self.U

    def select_direction(self,gradients):
        # solve quadratic programming problem for dual function

        n = len(gradients)
        x = cp.Variable(n)

        G = []

        # vectorize gradients matrices
        for i,g in enumerate(gradients):
            dim,n_components = g.shape
            vec = g.reshape(-1)
            G.append(vec)
        
        # pack into a single matrix
        G = np.stack(G,axis=1)
        P = G.T @ G
        objective = (1/2)*cp.quad_form(x,P)
        
        ones = np.ones(n)
        zeros = np.zeros(n)
        
        # constraints enforce probability simplex
        sum_to_one = ones.T @ x == 1
        non_negative = x >= zeros

        prob = cp.Problem(cp.Minimize(objective),[non_negative,sum_to_one])
        prob.solve()

        lambda_hat = x.value
        self.lambda_hat = lambda_hat
        direction = G @ self.lambda_hat
        direction = direction.reshape(dim,n_components)
        
        return -direction

    def orthogonal_projection(self,update):
        
        # use singular value decomposition to find orthogonal Procrustes problem
        u,s,vh = np.linalg.svd(update,full_matrices=False)
        orth_proj = u @ vh
       
        print("Loss of orthogonal",np.linalg.norm(update-orth_proj,ord='fro'))
        return orth_proj
