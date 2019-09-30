import numpy as np
import pandas as pd
import os
import sys
import torch.utils.data
import math
import pdb
from random import shuffle
import save_load_util as pa
from scipy.sparse import csr_matrix
import scipy


def format_data(args):

    #Import Preprocessed Data
    xtrain, ytrain, xtest, ytest, trainref, testref, refcols, featcols, labelcols, feathist = pa.load_all('xtrain.npz', 'xtest.npz', 'labels_and_references.npz', 'cols_and_hist.pkl',args['data_loc'])

    #Remove Columns that are zero in xtest
    loader = np.load(args['data_loc']+'xtrain_new.npz')
    xtrain=csr_matrix((loader['value'],(loader['rows'],loader['cols'])),shape=loader['shape'])

    #Remove Multitask Representation
    xtrain = xtrain[:,:len(featcols)]
    xtest = xtest[:,:len(featcols)]

    #Split Train into Validation (80/20) - keep patients together
    pid_train=trainref[:,refcols['PatientID']]
    np.random.seed(10)
    val_size = int(.2*len(np.unique(pid_train)))
    pid_val = np.random.choice(np.unique(pid_train),size=val_size,replace=False)
    # these are the same each time
    val_key = pd.DataFrame({'pid':pid_train})
    val_key['index'] = np.arange(val_key.shape[0])
    val_key = val_key.merge(pd.DataFrame({'pid':pid_val}),how='left',on='pid',indicator=True)
    val_key['val'] = val_key._merge=='both'

    #Sort by eid/day. Return key: DF with eid, seqlen, start_index
    day_train=trainref[val_key.loc[val_key.val==False,'index'],refcols['Day']]
    day_val=trainref[val_key.loc[val_key.val==True,'index'],refcols['Day']]
    day_test=testref[:,refcols['Day']]    
                       
    eid_train =trainref[val_key.loc[val_key.val==False,'index'],refcols['EncounterID']]
    eid_val =trainref[val_key.loc[val_key.val==True,'index'],refcols['EncounterID']]
    eid_test=testref[:,refcols['EncounterID']]
                       
    xtest, ytest, keytest = sort_eid_day(xtest, ytest, eid_test, day_test)
    xval, yval, keyval = sort_eid_day(xtrain[val_key.loc[val_key.val==True,'index'],:],
                                 ytrain[val_key.loc[val_key.val==True,'index']], eid_val, day_val)
    xtrain, ytrain, keytrain = sort_eid_day(xtrain, ytrain, eid_train, day_train)
    
    return xtrain, ytrain, keytrain, xval, yval, keyval, xtest, ytest, keytest

def sort_eid_day(data, labels, eid, day):
    key = pd.DataFrame({'eid':eid, 'day':day})
    key['index'] = np.arange(key.shape[0])
    key.sort_values(by=['eid','day'],inplace=True)
    
    data = data[key.index.values,:]
    labels = labels[key.index.values]
    key['new_index'] = np.arange(key.shape[0])
    key = key.groupby(['eid'], as_index=False).agg({'day':'count','new_index':'first'})
    key.rename(columns={'day':'seqlen','new_index':'start_index'}, inplace=True)
    return data, labels, key



class dataset_obj(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        
        self.args = args.copy()
        self.data = scipy.sparse.load_npz(os.path.join(args['save_folder'],args['id'],"data"+dataset+"x_temp.npz"))
        self.labels = np.load(os.path.join(args['save_folder'],args['id'],"data"+dataset+"y_temp.npy"))        
        self.datakey = pd.read_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),key=dataset)   
        
        self.N = self.datakey.shape[0]
        self.d = self.data.shape[1]
        
        self.labels = torch.LongTensor(self.labels)
        

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        start_index = self.datakey.iloc[idx,2] 
        seqlen = self.datakey.iloc[idx,1]
        x = csr_matrix.reshape(self.data[start_index:start_index+seqlen,:],(seqlen,self.d))
        y = self.labels[start_index:start_index+seqlen]
        return torch.FloatTensor(x.todense()), y, idx, seqlen

    
def custom_collate_fn(batch):
    # batch is a list of tuples where each tuple is the return of the dataset object
    # i.e. (data, label, idx)
    
    # https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

    data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch],batch_first=True)
    # default pads with 0. 
    # returns (max_seqlen, batch_size, d)
    labels = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch],batch_first=True)
    seqlen = [item[3] for item in batch]
    indices = [item[2] for item in batch]

    return [data, labels, seqlen, indices]    

