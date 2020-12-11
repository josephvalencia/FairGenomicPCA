import numpy as np
from sklearn import preprocessing
from src.data_utils import load_1KG_genotype, load_1KG_annotations
from src.PCA import frank_wolfe_NSW, loss_PCA, nash_social_welfare
import seaborn as sns
import matplotlib.pyplot as plt

def keep_top_variance(train_data,top_snps):

    variance = train_data.var(axis=0)
    idx = np.argsort(-variance)
    train_data = train_data[:,idx]
    train_data = train_data[:,:top_snps]

    return train_data

def keep_random(train_data,top_snps):

    n,m = train_data.shape
    idx = np.random.choice(np.arange(m),size=top_snps)
    train_data = train_data[:,idx]
    return train_data
    
if __name__ == "__main__":

    train_data = load_1KG_genotype("data/1000Genomes/recoded_1000G.noadmixed.mat")
    train_ids = load_1KG_annotations("data/1000Genomes/recoded_1000G.raw.noadmixed.ids")
    train_labels = load_1KG_annotations("data/1000Genomes/recoded_1000G.raw.noadmixed.lbls_super")

    le = preprocessing.LabelEncoder()
    racial_labels = le.fit_transform(train_labels).ravel()

    NUM_TOP_SNP = 1000
    train_data = keep_top_variance(train_data,NUM_TOP_SNP)

    train_racial_groups = {}

    for c in np.unique(racial_labels).tolist():
        index = racial_labels == c
        group_data = train_data[index,:]
        group_data = preprocessing.scale(group_data)
        train_racial_groups[c] = group_data
   
    train_scaled = preprocessing.scale(train_data)

    trial_num = 0
    NUM_COMPONENTS = 22

    princ_comp = frank_wolfe_NSW(train_scaled,train_racial_groups,d=NUM_COMPONENTS,learning_rate = 1e-1,epsilon=1e-4)
    
    savefile = "1000G_FW_{}_{}.npz".format(NUM_TOP_SNP, NUM_COMPONENTS)
   
    with open("1000G_FW_results.csv",'w') as outFile:

        for n in range(1,23):
            pc_top = princ_comp[:,:n]
            projected = train_scaled @ pc_top
            reconstruction_loss = loss_PCA(train_scaled,pc_top)
            nash = nash_social_welfare( 


    np.savez(savefile,proj=princ_comp)
    
    '''
    pc_top = princ_comp[:,:2]
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()

    a,b = np.split(projected,2,1)
    a = a.ravel()
    b = b.ravel()

    sns.scatterplot(x=a,y=b,hue=train_labels,ax=ax)
    plt.ylabel('PC 2')
    plt.xlabel('PC 1')
    plt.title('Nash Social Welfare')
    plt.savefig("NSW_scatter_full.png")
    plt.close()
    '''
