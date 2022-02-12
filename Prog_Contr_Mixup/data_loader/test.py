import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def process_adult_dataset(ds):
    train = pd.read_csv(ds)
    target = ' <=50K'

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object': # or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train[col].fillna(train.loc[:, col].mean(), inplace=True)
            


    features = [ col for col in train.columns if col not in [target]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    cont_idxs = list(set(range(14))-set(cat_idxs))
    
    for col_i in cont_idxs:
        train[train.columns[col_i]] = train[train.columns[col_i]] / train[train.columns[col_i]].max()


    target = train.pop(target)
    return train.to_numpy().astype(np.float32), target.to_numpy().astype(np.float32), cat_idxs, cat_dims, cont_idxs

dataset = 'data/census-income-train.csv'
process_adult_dataset(dataset)