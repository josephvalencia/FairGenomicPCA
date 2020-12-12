import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.svm import SVC
from src.data_utils import load_1KG_genotype , load_1KG_annotations , keep_top_variance 

def trial(data,labels,estimator,seed,num_classes):

    if estimator == "knn":
        model = KNeighborsClassifier(n_neighbors=8)
    elif estimator == "svm":
        model = SVC() 

    all_class_f1 = make_scorer(encode_multi_f1,average=None,labels=np.arange(num_classes))
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    encoded_scores = cross_val_score(model,X=data,y=labels,cv=kfold,scoring=all_class_f1)

    all_scores = []

    for s in encoded_scores.tolist():
        decoded = decode(s)
        all_scores.append(decoded)

    return  all_scores

def run_all_trials(data,labels,classes,estimator,projection):

    all_scores = []

    for i in range(10):
        results = trial(data,labels,estimator=estimator,seed=i,num_classes=len(classes))
        all_scores.extend(results)

    all_scores = np.vstack(all_scores)
    
    n_trials,n_classes = all_scores.shape

    mean = all_scores.mean(axis=0)
    std_err = all_scores.std(axis=0) / np.sqrt(n_trials)

    for name, c in classes.items():
        full_name = name.replace("_"," ")
        f1 = "{}+-{}".format(round(mean[c],3),round(std_err[c],3))
        entry = "{}\t{}\t{}\t{}".format(full_name,estimator,projection,f1)
        print(entry)

def decode(score):
    # decode encoded per-class F1

    code = str(score)
    if len(code) % 4 != 0:
        code = "0"+code
   
    decoded = []
    for i in range(0,len(code),4):
        word = int(code[i:i+4]) / 1000
        decoded.append(word)

    return np.asarray(decoded)

def encode_multi_f1(y,y_pred, **kwargs):
    # hack to encode per-class F1 as a single number
    
    all_classes = f1_score(y,y_pred,**kwargs)
    rounded = 1000 *all_classes
    rounded = rounded.astype(int)
    m = rounded.size
       
    encoding = ""
    for i in range(m):
        c = str(rounded[i])
        pad = 4 - len(c)
        if pad > 0:
            c = "0"*pad + c
        encoding +=c
    
    return int(encoding)

def make_labels(labels):

    le = preprocessing.LabelEncoder()
    labels_num = le.fit_transform(labels).ravel()
    mapping = dict(zip(le.classes_,le.transform(le.classes_)))

    return labels_num , mapping

if __name__ == "__main__":

    train_data = load_1KG_genotype("data/1000genomes/recoded_1000g.noadmixed.mat")
    micro_labels = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.lbls3_3")
    macro_labels = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.lbls_super")

    train_data = keep_top_variance(train_data,5000)
    scaled_data = preprocessing.scale(train_data)
    
    micro_labels_num, micro_mapping = make_labels(micro_labels)
    macro_labels_num, macro_mapping = make_labels(macro_labels)

    fw = np.load("1000G_FW_5000_22.npz",allow_pickle=True)['PC']
    vanilla = np.load("1000G_PCA_5000_22.npz",allow_pickle=True)['PC']

    top_fw = fw[:,:10]
    top_vanilla = vanilla[:,:10]

    data_fw = scaled_data @ top_fw
    data_vanilla = scaled_data @ top_vanilla
   
    print("Population\tProjection\tEstimator\tF1")
    run_all_trials(data_vanilla,micro_labels_num,classes=micro_mapping,estimator="svm",projection="Vanilla PCA")
    run_all_trials(data_fw,micro_labels_num,classes=micro_mapping,estimator="svm",projection="Frank-Wolfe-Nash")

    top_fw = fw[:,:2]
    top_vanilla = vanilla[:,:2]

    data_fw = scaled_data @ top_fw
    data_vanilla = scaled_data @ top_vanilla
   
    run_all_trials(data_vanilla,micro_labels_num,classes=micro_mapping,estimator="knn",projection="Vanilla PCA")
    run_all_trials(data_fw,micro_labels_num,classes=micro_mapping,estimator="knn",projection="Frank-Wolfe-Nash")

