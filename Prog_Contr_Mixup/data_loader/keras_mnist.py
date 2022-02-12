import numpy as np
import torch
#from tensorflow.keras.datasets import mnist
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy

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

def  calculate_representation_statistics(df,targets, cat_idxs):
    # df1 = pd.read_csv('dataset/dataset_shuffled_trfViol.csv')
    # df,targets = prepare_traffic_violation_dataset(df)
    df=df.reset_index()
    targets = targets.reset_index()
    del df['index']
    del targets['index']
    # df.reset_index()
    print("[INFO] - Generate representation ...")
    df['Label']=targets
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

METHODS = ['', 'supervised', 'semisupervised', 'pseudolabeling']

class KerasMNIST(Dataset):
    """ Implements keras MNIST torch.utils.data.Dataset


        Args:
            train (bool): flag to determine if train set or test set should be loaded
            labeled_ratio (float): fraction of train set to use as labeled
            method (str): valid methods are in `METHODS`.
                'supervised': getitem return x_labeled, y_label
                'semisupervised': getitem will return x_labeled, x_unlabeled, y_label
                'pseudolabeling': getitem will return x_labeled, x_unlabeled, y_label, y_pseudolabel
            random_seed (int): change random_seed to obtain different splits otherwise the fixed random_seed will be used

    """
    def __init__(self,
                 train: bool,
                 labeled_ratio: float=0.0,
                 method: str='supervised',
                 random_seed: int=None):
        super().__init__()
        print('Dataloader __getitem__ mode: {}'.format(method))
        assert method.lower() in METHODS , 'Method argument is invalid {}, must be in'.format(METHODS)
        self.repres ='condProb'
        # self.data_tr, self.target_tr, self.data_tst, self.target_tst, self.cat_idx, self.cont_idx =  process_adult_dataset('data/census-income-train.csv','data/census-income-test.csv')
        self.data_tr, self.target_tr, self.data_tst, self.target_tst, self.cat_idx, self.cont_idx = process_Viol_traffics_dataset('/data/dataset_trfVio.csv')

        if (train):
            self.data, self.targets = self.data_tr, self.target_tr
            # (self.data, self.targets (_, _) = mnist.load_data()
        else:
            # self.data, self.targets, _,_,_ = process_adult_dataset('data/census-income-test.csv',istrain=False)
            # (_, _), (self.data, self.targets) = mnist.load_data()
            self.data, self.targets = self.data_tst, self.target_tst

        # self.data = self.data / 255.0
        self.train = train
        # self.data = np.reshape(self.data, (len(self.targets), -1)).astype(np.float32)
        self.method = method.lower()
        idx = np.arange(len(self.targets))
        self.labeled_idx = idx
        self.unlabeled_idx = idx
        self.idx = idx
        if (labeled_ratio > 0):
            if random_seed is not None:
                idx = np.random.RandomState(seed=random_seed).permutation(len(self.targets))
            else:
                idx = np.random.permutation(len(self.targets))
            self.idx = idx
            if labeled_ratio <= 1.0:
                ns = labeled_ratio * len(self.idx)
            else:
                ns = labeled_ratio
            ns = int(ns)
            labeled_idx = self.idx[:ns]
            unlabeled_idx = self.idx[ns:]
            self.labeled_idx =  labeled_idx
            self.unlabeled_idx = unlabeled_idx
        if (train):
            self.cond_probs, self.num_classes = calculate_representation_statistics(self.data.iloc[labeled_idx], self.targets.iloc[labeled_idx], self.cat_idx)
            dict = {'cond_prob':self.cond_probs, 'num_classes':self.num_classes}
            np.save('data/cond_prob_w_pseudolabel.npy',dict)
        else:
            dict = np.load('data/cond_prob_w_pseudolabel.npy',allow_pickle=True)
            self.cond_probs = dict.item().get('cond_prob')
            self.num_classes = dict.item().get('num_classes')
        self.data = self.data.to_numpy()
        self.targets = self.targets.to_numpy()
        self._pseudo_labels = list()
        self._pseudo_labels_weights = list()
        self.refined_unlabeled_idx = list()
        self.agree=list()
        self.probs_lp = list()
        

    def get_pseudo_labels(self):
        return self._pseudo_labels

    def set_pseudo_labels(self, pseudo_labels):
        self._pseudo_labels = pseudo_labels

    def set_pseudo_labels_weights(self, pseudo_labels_weights):
        self._pseudo_labels_weights = pseudo_labels_weights
    
    def update_dataset_by_pseudo_labels(self):#,pseudo_labels_weights,lp_pseudo_labels,cls_pseudo_labels):
        '''
        arg9w_idx = np.where(self._pseudo_labels_weights>=0.9)[0]
        #final_idx = arg9w_idx[np.where(self.agree[arg9w_idx])[0]] #agreement_weight>0.9
        final_idx = arg9w_idx
        rep_gen_idx=final_idx #self.idx[self._pseudo_labels_weights>0]
        '''
        # l0=np.where(self.probs_lp[:,0]>0.8)[0]
        # l1=np.where(self.probs_lp[:,1]>0.8)[0]
        # lall=np.concatenate([l0,l1])
        # l=[]
        # for i in range(self.num_classes):
        #     l.append(np.where(self.probs_lp[:,i]>0.8)[0])

        # lall=np.concatenate(l)
        # rep_gen_idx = np.where(self.agree[lall])[0]
        # rep_gen_idx = np.unique(np.concatenate([rep_gen_idx,self.labeled_idx]))

        #self.cond_probs,_ = calculate_representation_statistics(self.data_tr.iloc[rep_gen_idx], pd.DataFrame(self._pseudo_labels[rep_gen_idx]),self.cat_idx)
        self.cond_probs,_ = calculate_representation_statistics(self.data_tr, pd.DataFrame(self._pseudo_labels),self.cat_idx)
        dict = {'cond_prob':self.cond_probs, 'num_classes':self.num_classes}
        np.save('data/cond_prob_w_pseudolabel.npy',dict)

        # tmp_weight = copy.copy(self._pseudo_labels_weights)
        # tmp_weight[self.labeled_idx]=0
        # self.refine_unlabeled_idxs = np.where(tmp_weight>0)[0]
    def update_agree(self, agr):
        self.agree = agr       

    def update_pseudo_labels_probs(self,probs):
        self.probs_lp = probs

    def __len__(self):
        if (self.method == 'pseudolabeling'):
            return len(self.idx)
        return len(self.labeled_idx)

    def _semisupervised__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        img = self.create_representation(img)
        uidx = np.random.randint(0, len(self.unlabeled_idx))
        uidx = self.unlabeled_idx[uidx]
        uimg = self.data[uidx]
        uimg = self.create_representation(uimg)
        if len(self.refined_unlabeled_idx):
            uidx = np.random.randint(0,len(self.refined_unlabeled_idx))#np.random.randint(0, len(self.unlabeled_idx))
            uidx = self.refined_unlabeled_idx[uidx]
            uimg = self.data[uidx]
            uimg = self.create_representation(uimg)
            
        # else:
        if len(self._pseudo_labels):
            utarget = self._pseudo_labels[uidx]
            uweight = self._pseudo_labels_weights[uidx]
            return img, uimg, target, utarget, uweight
            

        return img, uimg, target

    def _normal__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        img = self.create_representation(img)
        return img, target

    def _pseudolabeling__getitem__(self, idx):
        idx = self.idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        img = self.create_representation(img)
        labeled_mask = np.array([False], dtype=np.bool)
        if idx in self.labeled_idx:
            labeled_mask[0] = True
        idx = np.asarray([idx])
        return img, target, labeled_mask, idx

    def __getitem__(self, idx):
        if self.method == 'semisupervised' and self.train:
            return self._semisupervised__getitem__(idx)
        if self.method == 'pseudolabeling' and self.train:
            return self._pseudolabeling__getitem__(idx)
        else:
            return self._normal__getitem__(idx)

    def create_representation(self,x):
        if self.repres=='condProb':
            record = np.zeros(len(self.cat_idx)*self.num_classes+len(self.cont_idx),dtype=float)
            tmp_idx = 0
            for j in range(len(self.cat_idx)+len(self.cont_idx)):
                if self.cond_probs[j] == 'cont':
                    record[tmp_idx] = x[j]
                    tmp_idx+=1
                else:
                    try:
                        record[tmp_idx:tmp_idx+self.num_classes] = self.cond_probs[j][x[j]]# 2D output
                    except:
                        pass # leave zero
                    tmp_idx+=self.num_classes
        elif self.repres == 'onehot':
            # record=self.OHtransformer.transform(x)
            record = self.OHtransformer.transform(np.expand_dims(x,0)).toarray()
        return record.astype(np.float32)
