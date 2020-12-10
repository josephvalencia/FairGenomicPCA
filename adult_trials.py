import numpy as np
from sklearn import preprocessing
from src.PCA import run_PCA , FairPCA, frank_wolfe_NSW, factors

if __name__ == "__main__":

    train_npz = np.load("data/adult/adult.data.npz",allow_pickle=True)
    test_npz = np.load("data/adult/adult.test.npz",allow_pickle=True)

    train_data = train_npz['data']
    test_data = test_npz['data']

    train_scaled = preprocessing.scale(train_data)
    labels = train_npz['labels']
    racial_label_indices = np.char.startswith(labels,"race")
    racial_labels = np.nonzero(racial_label_indices)[0]

    train_racial_groups = {}
   
    for c in racial_labels.tolist():
        group_index = train_data[:,c] == 1
        group = train_scaled[group_index,:]
        train_racial_groups[c] = group
    
    d = 10
    X = frank_wolfe_NSW(train_scaled,train_racial_groups,d=d,learning_rate = 1e-1,epsilon=1e-5)
    P = factors(X,d)

    
