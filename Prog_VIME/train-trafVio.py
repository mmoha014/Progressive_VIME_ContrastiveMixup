from VIME import vime_semi, vime_self, mlp, perf_metric
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random as python_random
python_random.seed(123)
tf.random.set_seed(123) 
import warnings
warnings.filterwarnings('ignore')


def process_criteo_dataset():
    df = pd.read_csv('/home/morteza/Documents/Code Study/criteo/short_criteo.csv')
    df=df.drop([df.columns[0], df.columns[1]],axis=1)
    target = df.pop(df.columns[0])
    cont_idx=range(13)
    cont_cols = []
    cat_idx = range(13,39)
    for i in cont_idx:
        cont_cols = cont_cols +[df.columns[i]]
    # cont_cols = np.asarray(cont_cols)
    scaler = StandardScaler()
    cont_features = scaler.fit_transform(df[cont_cols])
    for i in cont_idx:
        df.iloc[:,i] = cont_features[:,i]
    
    for i in cat_idx:
        df.iloc[:,i] = df.iloc[:,i].astype(int)
    
    dftr, dftst, Ytr, Ytst = train_test_split(df,target,test_size=0.2,shuffle=False)
    return dftr, Ytr, dftst, Ytst, cat_idx, cont_idx

def process_drug_directory_dataset(adrs):
    df=pd.read_csv(adrs, sep='\t', encoding = "ISO-8859-1")
    target_col = 'PRODUCTTYPENAME'
    cols = ['PRODUCTID', 'PRODUCTNDC', 'PRODUCTTYPENAME', 'NONPROPRIETARYNAME',
        'DOSAGEFORMNAME', 'STARTMARKETINGDATE', 'ENDMARKETINGDATE',
        'MARKETINGCATEGORYNAME', 'LABELERNAME', 'SUBSTANCENAME',
        'ACTIVE_NUMERATOR_STRENGTH', 'ACTIVE_INGRED_UNIT', 'DEASCHEDULE',
        'LISTING_RECORD_CERTIFIED_THROUGH']

    le = LabelEncoder()
    scaler = StandardScaler()
    df=df.drop(['PRODUCTID', 'PRODUCTNDC','ACTIVE_NUMERATOR_STRENGTH','DEASCHEDULE'],axis=1)
    #===========================================================================
    df['PRODUCTTYPENAME'] = le.fit_transform(df['PRODUCTTYPENAME'].str.lower())
    df['NONPROPRIETARYNAME'] = le.fit_transform(df['NONPROPRIETARYNAME'].str.lower())
    df['DOSAGEFORMNAME'] = le.fit_transform(df['DOSAGEFORMNAME'].str.lower())
    # STARTMARKETINGDATE
    df['STARTMARKETINGDATE'] = pd.to_datetime(df['STARTMARKETINGDATE'], format='%Y%m%d')
    df['startMarketing_year'] = df['STARTMARKETINGDATE'].dt.year
    df['startMarketing_month'] = df['STARTMARKETINGDATE'].dt.month
    df['startMarketing_day'] = df['STARTMARKETINGDATE'].dt.day
    df = df.drop('STARTMARKETINGDATE',axis=1)
    # ENDMARKETINGDATE has 18963 nan values out of 19764
    df['ENDMARKETINGDATE'] = df['ENDMARKETINGDATE'].fillna(value=int(date.today().strftime('%Y%m%d')))
    df['ENDMARKETINGDATE'] = pd.to_datetime(df['ENDMARKETINGDATE'], format='%Y%m%d')
    df['endMarketing_year'] = df['ENDMARKETINGDATE'].dt.year
    df['endMarketing_month'] = df['ENDMARKETINGDATE'].dt.month
    df['endMarketing_day'] = df['ENDMARKETINGDATE'].dt.day
    df = df.drop('ENDMARKETINGDATE',axis=1)

    df['MARKETINGCATEGORYNAME'] = le.fit_transform(df['MARKETINGCATEGORYNAME'].str.lower())

    df['LABELERNAME'] = le.fit_transform(df['LABELERNAME'].str.lower())

    df['SUBSTANCENAME']= df['SUBSTANCENAME'].str.lower()
    df['SUBSTANCENAME'] = le.fit_transform(df['SUBSTANCENAME'].fillna(value='nan'))

    #-> numerical :df['ACTIVE_NUMERATOR_STRENGTH']
    df['ACTIVE_INGRED_UNIT'] = le.fit_transform(df['ACTIVE_INGRED_UNIT'].str.lower())
    df=df.drop(['LISTING_RECORD_CERTIFIED_THROUGH'],axis=1)

    cat_idx = [1,2,3,4,5,6]
    cont_idx = [7,8,9,10,11,12]
    newcols = df.columns
    if len(cont_idx)>0:
        cont_features = scaler.fit_transform(df[[newcols[7],newcols[8],newcols[9],newcols[10],newcols[11],newcols[12]]])
    for i,idx in enumerate(cont_idx):
        df[newcols[idx]] = cont_features[:,i]
    # cont_features = scaler.fit_transform(df[['startMarketing_year','startMarketing_month','startMarketing_day','endMarketing_year',
    #                     'endMarketing_month', 'startMarketing_day']])
    # for 
    # Date -> df['LISTING_RECORD_CERTIFIED_THROUGH']
    # df['LISTING_RECORD_CERTIFIED_THROUGH'] = df['LISTING_RECORD_CERTIFIED_THROUGH'].fillna(value=int(date.today().strftime('%Y%m%d')))
    # df['LISTING_RECORD_CERTIFIED_THROUGH'] = pd.to_datetime(df['LISTING_RECORD_CERTIFIED_THROUGH'], format='%Y%m%d')

    # df['certified_year'] = df['LISTING_RECORD_CERTIFIED_THROUGH'].dt.year
    # df['certified_month'] = df['LISTING_RECORD_CERTIFIED_THROUGH'].dt.month
    # df['certified_day'] = df['LISTING_RECORD_CERTIFIED_THROUGH'].dt.day
    target = df.pop(target_col)
    dftr, dftst, Ytr, Ytst = train_test_split(df,target,test_size=0.2,shuffle=False)
    return dftr, Ytr, dftst, Ytst, cat_idx, cont_idx

def process_Viol_traffics_dataset(ds_tr):
    train = pd.read_csv(ds_tr)
    target = 'Label'

    ytr = train.pop('Label')
    Xtr = train
       
    Xtr.pop('article')
    
    cols_nan = []
    search_cols = []
    for i in range(len(Xtr.columns)):
        if Xtr[Xtr.columns[i]].isna().sum()>0:
            if 'search' not in Xtr.columns[i]:
                cols_nan.append(Xtr.columns[i])
            else:
                search_cols.append(Xtr.columns[i])

    # drop search columnss
    for c in search_cols: 
        Xtr.pop(c)

    Xtr['violation_type'] = ytr
    Xtr.dropna(subset=cols_nan, inplace = True)
    Ytr = Xtr.pop('violation_type')
    Xtr.pop('agency')

    colss=np.array(Xtr.columns)
    
    colss = ['subagency', 'description', 'location', 'accident', 'belts',
        'personal_injury', 'property_damage', 'fatal', 'commercial_license',
        'hazmat', 'commercial_vehicle', 'alcohol', 'work_zone', 'state',
        'vehicletype', 'make', 'model', 'color', 'charge',
        'contributed_to_accident', 'race', 'gender', 'driver_city',
        'driver_state', 'dl_state', 'arrest_type']#colss[idx]
    dropCols = set(Xtr.columns)-set(colss)
    Xtr = Xtr.drop(dropCols,axis=1)
    df = pd.DataFrame()
    le = LabelEncoder()
    for c in colss:
        # if X[c].shape[]
        df[c] = le.fit_transform(Xtr[c])

    le = LabelEncoder()
    Ytr = le.fit_transform(Ytr)
    #df['Label']=Y
    Ytr = pd.DataFrame(Ytr)
    df = df.reset_index()
    df.pop('index')

    dftr, dftst, Ytr, Ytst = train_test_split(df,Ytr,test_size=0.2,shuffle=False)
    return dftr, Ytr, dftst, Ytst, np.arange(len(dftr.columns)), []
    # df_tr, target_tr,df_tst, target_tst, cat_idxs, cont_idxs
    

def  calculate_representation_statistics(df,targets, cat_idxs,num_classes=None):
    # df1 = pd.read_csv('dataset/dataset_shuffled_trfViol.csv')
    # df,targets = prepare_traffic_violation_dataset(df)
    df=df.reset_index()
    targets = targets.reset_index()
    del df['index']
    del targets['index']
    # df.reset_index()
    print("[INFO] - Generate representation ...")

    df['Label']=targets
    if num_classes==None:
        num_classes = int(targets.max()+1)
    total_rep = []
    cols_process = df.columns[:-1]
    for idx,col in enumerate(cols_process):
        if idx in cat_idxs:
            # pridfnt("Processing", col)
            df[col]=df[col].fillna('ffffffff')
            Nax = df.groupby(col).size()
            Naxc = df.groupby([col,'Label']).size()
            temp=[]
            #import pdb; pdb.set_trace()
            # print("[INFO] - Statistics calculation")
            t_rep = {}
            # t = tqdm(total=len(Nax),position=0)
            for i in range(len(Nax)):
                cat = Nax.index.values[i]
                v=Naxc[cat]

                # t=[[0,cat,lbl, num,Nax[i]] for num,lbl in zip(v,v.index.values)]
                repVec = np.zeros(num_classes, dtype=float)
                for num,lbl in zip(v,v.index.values):
                    repVec[int(lbl)] = num/Nax.values[i]

                t_rep[cat] = repVec
                # temp.append(t)
            total_rep.append(t_rep)
        else:
            total_rep.append('cont')

    return total_rep, num_classes

labeled_ratio=0.1
random_seed = 123
num_classes = 4
data_tr, target_tr, data_tst, target_tst, cat_idx, cont_idx = process_Viol_traffics_dataset('../traffic_violation/dataset_shuffled_trfViol.csv')
data, targets = data_tr, target_tr
idx = np.arange(len(targets))
labeled_idx = idx
unlabeled_idx = idx
idx = idx

if (labeled_ratio > 0):
    if random_seed is not None:
        idx = np.random.RandomState(seed=random_seed).permutation(len(targets))
    else:
        idx = np.random.permutation(len(targets))

# idx = np.arange(len(targets))
if labeled_ratio <= 1.0:
    ns = labeled_ratio * len(idx)
else:
    ns = labeled_ratio
    
ns = int(ns)
labeled_idx = idx[:ns]
valid_ns = int(len(labeled_idx)*0.1)
# valid_idx = labeled_idx[len(labeled_idx)-valid_ns:]
# labeled_idx = labeled_idx[:len(labeled_idx)-valid_ns]

unlabeled_idx = idx[ns:]
# labeled_idx =  labeled_idx
# unlabeled_idx = unlabeled_idx


cond_probs, num_classes = calculate_representation_statistics(data.iloc[labeled_idx], targets.iloc[labeled_idx], cat_idx)
dict1 = {'cond_prob':cond_probs, 'num_classes':num_classes}
np.save('data/cond_prob_w_pseudolabel.npy',dict1)



data_tr = data.to_numpy()
target_tr = targets.to_numpy()
data_ts = data_tst.to_numpy()
target_ts = target_tst.to_numpy()
pseudo_labels = list()
_pseudo_labels_weights = list()
refined_unlabeled_idx = list()
agree=list()
probs_lp = list()


def create_representation(x):
    record = np.zeros(len(cat_idx)*num_classes+len(cont_idx),dtype=float)
    tmp_idx = 0
    for j in range(len(cat_idx)+len(cont_idx)):
        if cond_probs[j] == 'cont':
            record[tmp_idx] = x[j]
            tmp_idx+=1
        else:
            try:
                record[tmp_idx:tmp_idx+num_classes] = cond_probs[j][x[j]]# 2D output
            except:
                pass # leave zero
            tmp_idx+=num_classes

    return record.astype(np.float32)

def update_representation():
    cpr_train = list()
    for i in range(len(data_tr)):
        cpr_train.append(create_representation(data_tr[i]))

    cpr_train = np.stack(cpr_train)

    cpr_test = list()
    for i in range(len(data_ts)):
        cpr_test.append(create_representation(data_ts[i]))
    cpr_test = np.stack(cpr_test)
    return cpr_train, cpr_test


def VIME_semi(x_train,y_train, x_unlab,x_test,y_test, epoch, total_train):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # MLP
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100

    if epoch==0:
        y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
        results = perf_metric('acc', y_test, y_test_hat)

        # Report performance
        print('Supervised Performance, Model Name: ' , 'MLP' , 
                ', Performance: ',results)

    p_m = 0.3
    alpha = 2.0
    K = 3
    beta = 1.0
    label_data_rate = 0.1
    file_name = './save_model/encoder_model.h5'
    vime_self_parameters = dict()
    vime_self_parameters['hidden_dim'] = 104
    vime_self_parameters['batch_size'] = 256
    vime_self_parameters['epochs'] = 15
    vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)

    # Save encoder
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    file_name = './save_model/trafVio_encoder_model222.h5'

    vime_self_encoder.save(file_name) 

    # Test VIME-Self
    x_train_hat = vime_self_encoder.predict(x_train)
    x_test_hat = vime_self_encoder.predict(x_test)

    y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
    results = perf_metric('acc', y_test, y_test_hat)

    print('VIME-Self Performance: ' + str(results)) 

    vime_semi_parameters = dict()
    vime_semi_parameters['hidden_dim'] = 104
    vime_semi_parameters['batch_size'] = 256
    vime_semi_parameters['iterations'] = 2000
    y_test_hat,pseudoLabels = vime_semi(x_train, y_train, x_unlab, x_test, 
                       vime_semi_parameters, p_m, K, beta, file_name,total_train)

    results = perf_metric('acc', y_test, y_test_hat)
    pl_acc =  perf_metric('acc', pseudoLabels, to_categorical(target_tr))
    

    print('VIME Performance: '+ str(results))
    print('PseudoLabel Performance: '+ str(pl_acc))
    
    return pseudoLabels

cpr_train, cpr_test = update_representation()
# import pdb; pdb.set_trace()
for epoch in range(8):
    if epoch==0:
        pseudoLabels = VIME_semi(cpr_train[labeled_idx],target_tr[labeled_idx],cpr_train[unlabeled_idx], cpr_test, target_ts, epoch,cpr_train)
    else:
        pseudoLabels = VIME_semi(cpr_train[labeled_idx],pseudoLabels[labeled_idx],cpr_train[unlabeled_idx], cpr_test, target_ts, epoch,cpr_train)
#     import pdb; pdb.set_trace()
#     pseudoLabels = np.argmax(pseudoLabels,axis=1)
#     pseudoLabels[labeled_idx] = target_tr[labeled_idx].squeeze()#tmp[:len(labeled_idx)]
#     # pseudoLabels[unlabeled_idx] = tmp[len(unlabeled_idx):]
#     cond_probs, num_classes = calculate_representation_statistics(data, pd.DataFrame(pseudoLabels), cat_idx)
#     cpr_train, cpr_test = update_representation()
    confidence = np.max(pseudoLabels,1)
    confidence[labeled_idx] = 1.0
    idxs = np.where(confidence>0.7)[0]
    pseudoLabels = np.argmax(pseudoLabels,axis=1)
    pseudoLabels[labeled_idx] = target_tr[labeled_idx].squeeze()#tmp[:len(labeled_idx)]
    # pseudoLabels[unlabeled_idx] = tmp[len(unlabeled_idx):]
    cond_probs, num_classes = calculate_representation_statistics(data.iloc[idxs], pd.DataFrame(pseudoLabels[idxs]), cat_idx)
    cpr_train, cpr_test = update_representation()