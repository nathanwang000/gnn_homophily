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

    #Sort by eid/day. Return key: DF with eid, seqlen, start_index
    day_train=trainref[:,refcols['Day']]
    day_test=testref[:,refcols['Day']]
    eid_train=trainref[:,refcols['EncounterID']]
    eid_test=testref[:,refcols['EncounterID']]
    xtest, ytest, keytest = sort_eid_day(xtest, ytest, eid_test, day_test)
    xtrain, ytrain, keytrain = sort_eid_day(xtrain, ytrain, eid_train, day_train)
    
    #Turn CSR -> COO
#     xtrain = xtrain.tocoo()
#     xtest = xtest.tocoo()
    return xtrain, ytrain, keytrain, xtest, ytest, keytest

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
        
#         self.data = torch.sparse.FloatTensor(torch.LongTensor(np.stack([self.data.row,self.data.col])), 
#                                             torch.LongTensor(self.data.data), 
#                                             torch.Size(self.data.shape))
        self.labels = torch.FloatTensor(self.labels)
        

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

    data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch])
    # default pads with 0. 
    # returns (max_seqlen, batch_size, d)
    labels = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch])
    seqlen = [item[3] for item in batch]

    data = torch.nn.utils.rnn.pack_padded_sequence(data,seqlen,enforce_sorted=False)
    labels = torch.nn.utils.rnn.pack_padded_sequence(labels,seqlen,enforce_sorted=False)
    # returns torch.nn.utils.rnn.PackedSequence, a named tuple (data, batch_sizes, sorted_indices, unsorted_indices)
    return [data, labels]

class BySequenceLengthSampler(torch.utils.data.sampler.Sampler):
    ''' 
    Batch Sampler: https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c
    Must provide bucket boundaries for data
        eg: bb = [50, 100, 125, 150, 175, 200, 250, 300]
    Groups samples into buckets by sequence length
    Shuffles them within bucketts
    '''

    def __init__(self, data_source, bucket_boundaries, batch_size=64,):
        #Determine length of each sample
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[0]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
        
    def __iter__(self):
        data_buckets = dict()
        # Assign each sample to a bucket
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        # Shuffle the data in these buckets
        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


    
    
    
    
    

