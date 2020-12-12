import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from src.data_utils import load_1KG_genotype, load_1KG_annotations, keep_random, keep_top_variance
from src.PCA import FrankWolfeNash, loss_PCA, nash_social_welfare
import seaborn as sns
import matplotlib.pyplot as plt

def summarize(save_prefix,PC,corr_list,train_scaled,n_components):

    with open(save_prefix+".csv",'w') as outFile:
        
        outFile.write("n_components\treconstruction_loss\tnash_social_welfare\n")

        for n in range(1,n_components+1):
            pc_top = PC[:,:n]
            projected = train_scaled @ pc_top
            loss = loss_PCA(train_scaled,pc_top)
            X = pc_top @ pc_top.T
            nash = nash_social_welfare(corr_list,X)
            entry ="{}\t{}\t{}\n".format(n,loss,nash)
            outFile.write(entry+"\n")

    np.savez(save_prefix+".npz",PC=PC)

if __name__ == "__main__":

    train_data = load_1KG_genotype("data/1000genomes/recoded_1000g.noadmixed.mat")
    train_ids = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.ids")
    train_labels = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.lbls_super")

    le = preprocessing.LabelEncoder()
    racial_labels = le.fit_transform(train_labels).ravel()
    
    '''
    NUM_TOP_SNP = 500
    train_data = keep_top_variance(train_data,NUM_TOP_SNP)
    '''
    
    train_racial_groups = {}

    for c in np.unique(racial_labels).tolist():
        index = racial_labels == c
        group_data = train_data[index,:]
        group_data = preprocessing.scale(group_data)
        train_racial_groups[c] = group_data
   
    train_scaled = preprocessing.scale(train_data)

    NUM_COMPONENTS = 22
    
    '''
    fwn = FrankWolfeNash(NUM_COMPONENTS)
    fwn.run(train_scaled,train_racial_groups,learning_rate = 1e-1,epsilon=1e-4)
  
    # extract principal components and correlation matrices
    PC = fwn.PC
    corr_list = fwn.corr_list

    save_prefix = "1000G_FW_{}_{}".format("ALL", NUM_COMPONENTS)
    summarize(save_prefix,PC,corr_list,train_scaled,NUM_COMPONENTS)
    
    corr_list = [ a.T @ a for a in train_racial_groups.values()]

    vanilla_pca = PCA(n_components=NUM_COMPONENTS)
    vanilla_pca.fit(train_scaled)
    PC = vanilla_pca.components_.T
    
    save_prefix = "1000G_PCA_{}_{}".format("ALL", NUM_COMPONENTS)
    summarize(save_prefix,PC,corr_list,train_scaled,NUM_COMPONENTS)
    '''

    pareto = ParetoPCA
