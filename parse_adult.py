import pandas as pd
from sklearn import preprocessing
import sys
import numpy as np

columns = ['age','workclass','fnlwgt','education','education-num','marital-stats','occupation','relationship',\
        'race','sex','capital-gain','capital-loss','hours-per-week','native-country','income-bracket']

non_numeric = ['workclass','education','marital-stats','occupation','relationship','race','sex']

df_train = pd.read_csv(sys.argv[1],names=columns)
df_test = pd.read_csv(sys.argv[2],names=columns,skiprows=1)

train_storage = []
test_storage = []

labels = []

categorical = True

for c in df_train.columns:

    train_raw = df_train[c].values.reshape(-1,1)
    test_raw = df_test[c].values.reshape(-1,1)
    
    if c in non_numeric:

        if categorical: 
            one_hot = preprocessing.OneHotEncoder()

            combined = np.concatenate([train_raw,test_raw],axis=0)
            one_hot.fit(combined)

            new_labels = ["{}_{}".format(c,x.lstrip()) for x in list(one_hot.categories_[0])]
            labels.extend(new_labels)

            train_dummy = one_hot.transform(train_raw).toarray()
            test_dummy = one_hot.transform(test_raw).toarray()

            train_storage.append(train_dummy)
            test_storage.append(test_dummy)
        else:
            le = preprocessing.LabelEncoder()

            combined = np.concatenate([train_raw,test_raw],axis=0)
            le.fit(combined)

            new_labels = ["{}_{}".format(c,x.lstrip()) for x in list(le.classes_[0])]
            labels.extend(new_labels)

            train_cat = le.transform(train_raw).reshape(-1,1)
            test_cat = le.transform(test_raw).reshape(-1,1)

            train_storage.append(train_cat)
            test_storage.append(test_cat)

    elif c != 'income-bracket' and c!= 'native-country':
        train_storage.append(train_raw)
        test_storage.append(test_raw)
        labels.append(c)

labels = np.asarray(labels)
test_numerical = np.concatenate(test_storage,axis=1)
train_numerical = np.concatenate(train_storage,axis=1)

np.savez("adult.test.npz",data=test_numerical,labels=labels)
np.savez("adult.data.npz",data=train_numerical,labels=labels)

