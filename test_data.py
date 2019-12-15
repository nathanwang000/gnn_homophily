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
import itertools
import datetime
import torch.nn.functional as F


def import_loc_aux_data(args):
    # Location DF #################################################################
    print('Importing Location Data...')
    # Bring in Location Data
    loc_df = pd.read_csv(args['locdata_loc'],sep='|')

    # Remove if..
    loc_df = loc_df.loc[loc_df.LocationCode==loc_df.LocationCode,:]
    loc_df = loc_df.loc[loc_df.LocationType==loc_df.LocationType,:]
    loc_df = loc_df.loc[loc_df.StartDate!=loc_df.EndDate,:]
    loc_df = loc_df.loc[loc_df.EndDate==loc_df.EndDate,:]
    loc_df = loc_df.loc[loc_df.StartDate==loc_df.StartDate,:]
    loc_df = loc_df.drop(['EncounterID'], axis=1)
    loc_df = loc_df.drop_duplicates()
    loc_df = loc_df.loc[loc_df.LocationType!='HS',:]

    # Remove Duplicates
    loc_df['duplicate'] = loc_df.duplicated(subset=['PatientID','StartDate','EndDate','LocationType'],keep=False)
    print('Number of Duplicate Rows Dropped: {}'.format(loc_df.duplicate.sum()))
    print('Total Number of Rows in DF: {}'.format(loc_df.shape[0]))
    # loc_df.loc[loc_df.duplicate==True,:]
    loc_df = loc_df.loc[loc_df.duplicate==False,:]

    # Reshape Wide
    loc_df = loc_df.set_index(["PatientID","AdmitDate",'DischargeDate',"StartDate", "EndDate","LocationType"])['LocationCode'].unstack()
    loc_df.drop(columns=['Bed','Room'], inplace=True)
    loc_df.reset_index(inplace=True)

    loc_df['StartDate'] = pd.to_datetime(loc_df.StartDate)
    loc_df['EndDate'] = pd.to_datetime(loc_df.EndDate)
    
    loc_df['AdmitDate'] = pd.to_datetime(pd.to_datetime(loc_df.AdmitDate).dt.date)
    loc_df['DischargeDate'] = pd.to_datetime(pd.to_datetime(loc_df.DischargeDate).dt.date)

    # Location IDs, Admit IDs
#     loc_df['locid'] = loc_df.fillna('MISSING').groupby(['Bed','FacilityCode','Room','Unit']).ngroup()
    loc_df['locid'] = loc_df.fillna('MISSING').groupby(['FacilityCode','Unit']).ngroup()
    loc_df['aid'] = loc_df.fillna('MISSING').groupby(['PatientID','AdmitDate','DischargeDate']).ngroup()

    # Auxiliary Information DF #########################################################################
    print('Importing Auxiliary Information...')
    
    # Bring in auxiliary information
    cdi_df = pd.read_csv(args['auxdata_loc'],sep='|')
    cdi_df.drop(columns='EncounterID',inplace=True)
    cdi_df.rename(columns={'Date':'OutcomeDate'},inplace=True)
    cdi_df['OutcomeDate'] = pd.to_datetime(cdi_df.OutcomeDate)
    cdi_df['AdmitDate'] = pd.to_datetime(pd.to_datetime(cdi_df.AdmitDate).dt.date)
    cdi_df['DischargeDate'] = pd.to_datetime(pd.to_datetime(cdi_df.DischargeDate).dt.date)

    # Admit ID
    cdi_df = cdi_df.merge(loc_df.loc[:,['PatientID','AdmitDate','DischargeDate','aid']].drop_duplicates(), how='left', on=['PatientID','AdmitDate','DischargeDate'],
                         indicator=True)
    print('Admit ID merge (dropped if not both)')
    print(cdi_df._merge.value_counts())
    cdi_df = cdi_df.loc[cdi_df._merge=='both',:]
    cdi_df.drop(columns=['_merge'], inplace=True)

    return loc_df, cdi_df

def format_data(args):
    
    #Import Preprocessed Data
    # xtrain, ytrain, xtest, ytest, trainref, testref, refcols, featcols, labelcols, feathist = pa.load_all('xtrain.npz', 'xtest.npz', 'labels_and_references.npz', 'cols_and_hist.pkl',args['data_loc'])
    _, _, xtest, ytest, _, testref, refcols, featcols, _, _ = pa.load_all('xtrain.npz', 'xtest.npz', 'labels_and_references.npz', 'cols_and_hist.pkl',args['data_loc'])

    #Remove Columns that are zero in xtest
#     loader = np.load(args['data_loc']+'xtrain_new.npz')
#     xtrain=csr_matrix((loader['value'],(loader['rows'],loader['cols'])),shape=loader['shape'])

    #Remove Multitask Representation
    # xtrain = xtrain[:,:len(featcols)]
    xtest = xtest[:,:len(featcols)]
    
    #Import location and auxiliary information
    loc_df, cdi_df = import_loc_aux_data(args)
    
    #Only keep patients with some location information. Merge on AdmitID
    key = pd.DataFrame({'AdmitDate':testref[:,refcols['AdmitDate']],
                       'DischargeDate':testref[:,refcols['DischargeDate']],
                       'PatientID':testref[:,refcols['PatientID']],
                       'eid':testref[:,refcols['EncounterID']],
                       'day':testref[:,refcols['Day']],
                       'Date':testref[:,refcols['Date']],
                       'y':ytest})
    key['index'] = np.arange(key.shape[0])
    key = key.merge(loc_df.loc[:,['PatientID','AdmitDate','DischargeDate','aid']].drop_duplicates(), 
                    how='inner', on=['PatientID','AdmitDate','DischargeDate'])
    
    # NEW LABEL: How many neighbors up to today?
    if args['task'] in ['neighbors','neighbor_deciles','unique_neighbors']:
        # Which locations have has a patient been in? (on a aid/Date basis)
        newlabel = key.loc[:,['aid','Date','index']].merge(loc_df.loc[:,['StartDate','EndDate','locid','aid']], how='left', on=['aid'])
        newlabel = newlabel.loc[((newlabel.Date>=newlabel.StartDate) & (newlabel.Date<=newlabel.EndDate)),:]
        newlabel.drop(['StartDate','EndDate'],axis=1,inplace=True)
        # Which other patients were in that location on that day? 
        newlabel = newlabel.merge(newlabel.drop(['index'],axis=1),how='left',on=['Date','locid'])
        newlabel = newlabel.loc[newlabel.aid_x!=newlabel.aid_y,:]
        newlabel = newlabel.drop(['locid'],axis=1).drop_duplicates() 
            # this is just in case so we don't double count neighbors, regardless of location
            #eg: may have met patient in loc a and b -> but only counts neighbor once. this is to reflect how the adj matrix is created.
        # Unique Neighbors
        if args['task'] == 'unique_neighbors':
            newlabel = newlabel.sort_values(['aid_x','Date','aid_y']).drop_duplicates(['aid_x','aid_y'], keep='first')
        # Collapse sum: how many neighbors per day.
        newlabel = newlabel.groupby(['aid_x','Date','index'],as_index=False)['aid_y'].count()
        # Insert Days with zero neighbors.
        newlabel = newlabel.merge(key.loc[:,['index','aid','Date']].rename(columns={'aid':'aid_x'}),how='outer',on=['index','aid_x','Date'],indicator=True)
        newlabel['aid_y'].fillna(value=0,inplace=True)
        newlabel.drop(['_merge'],axis=1,inplace=True)
        # Cumulative sum: how many neighbors up to now
        newlabel['y'] = newlabel.sort_values(['aid_x','Date']).groupby(['aid_x'],as_index=False)['aid_y'].cumsum().astype('float')
        # Merge new labels back into key
        key.drop(['y'],axis=1,inplace=True)
        key = key.merge(newlabel.loc[:,['index','y']],how='left',on='index')
        key['y'] = key['y']/100
        
    #Split Train/Val/Test
    key['val'] = key.AdmitDate.dt.month==3
    key['test'] = key.AdmitDate.dt.month==5
    key['train'] = key.AdmitDate.dt.month==4

    
    #neighbor_deciles Label
    if args['task'] == 'neighbor_deciles':
        bincuts = []
        for x in range(0,101,10):
            bincuts.append(np.percentile(key.loc[key['train']==True,'y'],x))
        bincuts[-1] = key.y.max()
        bincuts = np.sort(list(set(bincuts)))
        key['y'] = pd.cut(key['y'], bincuts, labels=np.arange(len(bincuts)-1), right=False)
        args['num_classes'] = len(bincuts)-1

    #Sort by eid/day. Add to key: eid, seqlen, start_index 
    xval, yval, keyval = sort_eid_day(xtest[key.loc[key.val,'index'],:],
                                      key.loc[key.val,'y'].values,
                                     key.loc[key.val,['eid','day','aid','AdmitDate','DischargeDate']])
    xtrain, ytrain, keytrain = sort_eid_day(xtest[key.loc[key.train,'index'],:], 
                                            key.loc[key.train,'y'].values,
                                            key.loc[key.train,['eid','day','aid','AdmitDate','DischargeDate']])
    xtest, ytest, keytest = sort_eid_day(xtest[key.loc[key.test,'index'],:], 
                                         key.loc[key.test,'y'].values,
                                         key.loc[key.test,['eid','day','aid','AdmitDate','DischargeDate']])

    return xtrain, ytrain, keytrain, xval, yval, keyval, xtest, ytest, keytest, loc_df, cdi_df

def sort_eid_day(data, labels, key):
    key.reset_index(inplace=True)
    key['index'] = np.arange(key.shape[0])
    key.sort_values(by=['eid','day'],inplace=True)
    
    data = data[key.index.values,:]
    labels = labels[key.index.values]
    key['new_index'] = np.arange(key.shape[0])
    key = key.groupby(['eid'], as_index=False).agg({'day':'count','new_index':'first',
                                                    'aid':'first','AdmitDate':'first',
                                                   'DischargeDate':'first'})
    key.rename(columns={'day':'seqlen','new_index':'start_index'}, inplace=True)
    return data, labels, key

def create_adj_mat(df, date, aid):
    
    df = df.loc[(df.StartDate<=date)&(df.EndDate>=date),['aid','locid']].drop_duplicates()
    df = df.groupby('locid',as_index=False).apply(lambda x: list(itertools.permutations(x.aid.values,2)))
    df = list(itertools.chain.from_iterable(df))
    row, col = map(list, zip(*df))
    df = pd.DataFrame({'row':row, 'col':col})
    
    key = df.loc[df.row!=aid,'row'].drop_duplicates()
    key = pd.concat([pd.Series(aid), key])
    n = len(key)
    key = pd.DataFrame({'aid':key, 'matid':np.arange(n)})
    adj_mat = np.identity(n)
    df = df.merge(key, how='left', left_on='row', right_on='aid')
    df = df.merge(key, how='left', left_on='col', right_on='aid')
    df.rename(columns={'matid_x':'row_matid','matid_y':'col_matid'}, inplace=True)
    adj_mat[df.row_matid,df.col_matid]=1        
    return adj_mat, key

def create_signal_mat(args, df, date, key):
#     T=args['sig_T']
#     df = df.loc[(df.OutcomeDate<=date)&(df.OutcomeDate>=date-datetime.timedelta(days=2*T-1)),
#                                         ['aid','OutcomeDate','Outcome']]
#     df = df.merge(key, how='inner', on='aid')    
#     df['daysfromnow'] = (date - df['OutcomeDate']).dt.days + 1 
#     df = df.loc[:,['matid','daysfromnow']].drop_duplicates()

#     sig_mat = np.zeros((key.shape[0],2*T))
#     sig_mat[df.matid.values,df.daysfromnow.values] = 1
#     sig_mat = np.concatenate((sig_mat,np.ones((sig_mat.shape[0],1))),axis=1)
    
#     return sig_mat

    return np.ones((key.shape[0],1))

def create_adj_mat_v2(df, date, aid, AdmitDate):
    #Created in Untitled7.ipynb
    
    # Keep if a neighbor of aid during date range (AdmitDate,Date)   
    df = df.loc[(df.StartDate<=date)&(df.EndDate>=AdmitDate),['aid','locid','StartDate','EndDate']]
    neighbors = df.loc[df.aid==aid,:].merge(df.loc[df.aid!=aid,:], how='left', on='locid')
    neighbors = neighbors.loc[(neighbors.StartDate_y<=neighbors.EndDate_x) & (neighbors.EndDate_y>=neighbors.StartDate_x),:]

    # Expand dates between Admit and date
    neighbors['date'] = date
    neighbors['Start'] = neighbors.loc[:,['StartDate_x','StartDate_y']].max(axis=1)
    neighbors['End'] = neighbors.loc[:,['date','EndDate_x','EndDate_y']].min(axis=1)
    if neighbors.shape[0]>0:
        neighbors = pd.concat([pd.DataFrame({'date': pd.date_range(row.Start, row.End, normalize=True),
                                             'aid':row.aid_x, 'neighbor_aid':row.aid_y,
                                             'Start':row.Start, 'End':row.End},
                                            columns = ['date','aid','neighbor_aid','Start','End'])
                               for i, row in neighbors.iterrows()], ignore_index=True).drop_duplicates()
        
        #Count Dates
        neighbors = neighbors.groupby('neighbor_aid', as_index=False)['date'].count()

    # Reinsert Self Relationship at Top // aid of interest on top
    neighbors = pd.concat([pd.DataFrame({'neighbor_aid':[aid], 'date':[(date-AdmitDate).days+1]}), neighbors], ignore_index=True, sort=False)
    
    #Recopy original adj matrix - make sure all functionality is moved over
    # Original adj matrix makes a full adj matrix - wrt all nodes 
    # Note - in this version - we ae not creating a full adj matrix wrt to all nodes. Only the first row is populated.
    n = neighbors.shape[0]
    neighbors['matid'] = np.arange(neighbors.shape[0])
    adj_mat = np.zeros((n,n))
    adj_mat[0,neighbors['matid']] = neighbors.date.values
    neighbors = neighbors.loc[:,['neighbor_aid','matid']]
    neighbors.rename({'neighbor_aid':'aid'},axis=1,inplace=True)

    return adj_mat, neighbors


class dataset_obj(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        
        self.args = args.copy()
        self.data = scipy.sparse.load_npz(os.path.join(args['save_folder'],args['id'],"data"+dataset+"x_temp.npz"))
        
        if self.args['use_cuda']: self.data = torch.FloatTensor(self.data.todense()).cuda()
        else: self.data = torch.FloatTensor(self.data.todense())
            
        if self.args['classification']:
            if self.args['use_cuda']: self.labels = torch.LongTensor(np.load(os.path.join(args['save_folder'],args['id'],"data"+dataset+"y_temp.npy"))).cuda()
            else: self.labels = torch.LongTensor(np.load(os.path.join(args['save_folder'],args['id'],"data"+dataset+"y_temp.npy")))
        else: 
            if self.args['use_cuda']: self.labels = torch.FloatTensor(np.load(os.path.join(args['save_folder'],args['id'],"data"+dataset+"y_temp.npy"))).cuda()
            else: self.labels = torch.FloatTensor(np.load(os.path.join(args['save_folder'],args['id'],"data"+dataset+"y_temp.npy")))
        self.datakey = pd.read_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),key=dataset)   
        self.loc_df = pd.read_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),key='loc_df')
        self.cdi_df = pd.read_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),key='cdi_df')
        
        self.N = self.datakey.shape[0]
        self.d = self.data.shape[1]
        
#         self.labels = torch.LongTensor(self.labels)        

    def __len__(self):
        return self.N
     
    def __getitem__(self,idx):
        
        start_index = self.datakey.iloc[idx,self.datakey.columns.get_loc('start_index')] 
        seqlen = self.datakey.iloc[idx,self.datakey.columns.get_loc('seqlen')]
        
#         x = csr_matrix.reshape(self.data[start_index:start_index+seqlen,:],(seqlen,self.d))
        x = self.data[start_index:start_index+seqlen,:].view(seqlen,self.d).contiguous()
        y = self.labels[start_index:start_index+seqlen].contiguous()
        
        AdmitDate = self.datakey.iloc[idx,self.datakey.columns.get_loc('AdmitDate')]
        DischargeDate = self.datakey.iloc[idx,self.datakey.columns.get_loc('DischargeDate')]
        aid = self.datakey.iloc[idx,self.datakey.columns.get_loc('aid')]
        A_list=[]
        S_list=[]

        for date in (AdmitDate + datetime.timedelta(n) for n in range(seqlen)):
            if self.args['tempjob'] == 'unique_neighbors_adj_mat':
                A_mat, key = create_adj_mat_v2(self.loc_df, date, aid, AdmitDate)
            else:
                A_mat, key = create_adj_mat(self.loc_df, date, aid)
            
            if self.args['use_cuda']:
                A_list.append(torch.FloatTensor(A_mat).cuda())
                S_list.append(torch.FloatTensor(create_signal_mat(self.args, self.cdi_df, date, key)).cuda())
            else: 
                A_list.append(torch.FloatTensor(A_mat))
                S_list.append(torch.FloatTensor(create_signal_mat(self.args, self.cdi_df, date, key)))
#         return torch.FloatTensor(x.todense()).cuda(), torch.LongTensor(y).cuda(), idx, seqlen, A_list, S_list
        return x, y, idx, seqlen, A_list, S_list
                                 
                                 
def mat_pad(batch_list, n, m):
    # If Adjmat, input is shape: [batch_size][T_i](n_i, n_i) 
    # If Sigmat, input is shape: [batch_size][T_i](n_i, 2*sig_T)
    # Returns a tensor of size (sum_i (T_i), n, m)
    # Removes the list of list format and stacks all matricies. 
    # Pads all adj/sig matricies so that they are shape (n, m)
    new_mat = [F.pad(timestep,(0,m-timestep.shape[1],0,n-timestep.shape[0]),value=0) for patient in batch_list for timestep in patient]
    new_mat = torch.stack(new_mat,dim=0)
    return new_mat

def custom_collate_fn(batch):
    # batch is a list of tuples where each tuple is the return of the dataset object
    # i.e. (data, label, idx)
    
    # https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

    data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch],batch_first=True)
    # default pads with 0. 
    # returns (max_seqlen, batch_size, d)
    labels = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch],batch_first=True)
    n_max = max([item[4][0].shape[0] for item in batch])
    
    A = mat_pad([item[4] for item in batch], n_max, n_max)
    S = mat_pad([item[5] for item in batch], n_max, batch[0][5][0].shape[1]) #first person, 5th item (S), first timestep
    
    if torch.cuda.is_available(): seqlen = torch.FloatTensor([item[3] for item in batch]).cuda()
    else: seqlen = torch.FloatTensor([item[3] for item in batch])
    indices = [item[2] for item in batch]

    return [data, labels, seqlen, indices, A, S]    
 

