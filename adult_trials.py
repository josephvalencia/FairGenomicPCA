import numpy as np
from sklearn import preprocessing
from src.PCA import PCA, FrankWolfeNash, ParetoPCA, loss_PCA, nash_social_welfare
#from sklearn.decomposition import PCA

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

    NUM_COMPONENTS = 22

    '''
    fwn = FrankWolfeNash(NUM_COMPONENTS)
    fwn.run(train_scaled,train_racial_groups,learning_rate = 1e-1,epsilon=1e-4)
  
    # extract principal components and correlation matrices
    PC = fwn.PC
    corr_list = fwn.corr_list

    save_prefix = "ADULT_FW_{}".format(NUM_COMPONENTS)
    summarize(save_prefix,PC,corr_list,train_scaled,NUM_COMPONENTS)

    vanilla_pca = PCA(n_components=NUM_COMPONENTS)
    vanilla_pca.run(train_scaled)
    #PC = vanilla_pca.components_.T
    PC = vanilla_pca.PC

    save_prefix = "ADULT_PCA_{}".format( NUM_COMPONENTS)
    summarize(save_prefix,PC,corr_list,train_scaled,NUM_COMPONENTS)

   '''

    pareto = ParetoPCA(NUM_COMPONENTS)
    pareto.run(train_scaled,train_racial_groups,max_iter=10000)


