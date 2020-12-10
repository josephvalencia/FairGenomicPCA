import numpy as np
from sklearn import preprocessing
from src.data_utils import load_1KG_genotype, load_1KG_annotations
from src.PCA import frank_wolfe_NSW, factors

def keep_top_variance(train_data,top_snps):

    top_snps = 5000

    variance = train_data.var(axis=0)
    idx = np.argsort(-variance)
    
    train_data = train_data[:,idx]
    train_data = train_data[:,:top_snps]

    return train_data 

if __name__ == "__main__":

    train_data = load_1KG_genotype("data/1000Genomes/recoded_1000G.noadmixed.mat")
    train_ids = load_1KG_annotations("data/1000Genomes/recoded_1000G.raw.noadmixed.ids")
    train_labels = load_1KG_annotations("data/1000Genomes/recoded_1000G.raw.noadmixed.lbls_super")

    le = preprocessing.LabelEncoder()
    racial_labels = le.fit_transform(train_labels).ravel()

    train_data = keep_top_variance(train_data,5000)

    train_racial_groups = {}

    for c in np.unique(racial_labels).tolist()[:1]:
        index = racial_labels == c
        print(index.shape)
        group_data = train_data[index,:]
        print(group_data.shape)
        group_data = preprocessing.scale(group_data)
        train_racial_groups[c] = group_data
   
    train_scaled = preprocessing.scale(train_data)

    d = 10
    X = frank_wolfe_NSW(train_scaled,train_racial_groups,d=d,learning_rate = 1e-1,epsilon=1e-5)
    P = factors(X,d)


