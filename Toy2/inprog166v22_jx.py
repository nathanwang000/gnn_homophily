import networkx as nx
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import seaborn as sns
import os
import scipy.sparse as sp
import copy
from sklearn.metrics import roc_auc_score



def return_file_name():
    return __file__


## FIX/FORMAT DATA
#####################################################################################################################

def new_location_codes(loc):
    '''Recalculate the LocationCodes location codes that have been properly aggregated'''
    
    # Unit Location Codes 
    loc = loc[['PatientID','Date','FacilityCode','Unit']].drop_duplicates()
    
    # Unit name is not unique, have to combine with facility code.
    # -- Checked process_top_features, it only splits on the first '_'
#     if loc.duplicated(['PatientID','Date','Unit'],keep=False).sum()>0: raise ValueError('Unit name is not unique')
    loc['LocationCode'] = [str(x)+'_'+str(y) for x,y in zip(loc.FacilityCode.values, loc.Unit.values)]
    unique_location_codes = loc.LocationCode.unique()
    
    # Groupby, comma delim list
    loc = loc.groupby(['PatientID','Date'], as_index=False)['LocationCode'].apply(', '.join).reset_index()
#     display(loc.head())
    loc.rename({0:'LocationCode'},axis=1,inplace=True)
    
    return loc, unique_location_codes


def calc_CP(loc, df, cap=14):
    '''Recalculate the colonization pressure using location codes that have been properly aggregated'''
    
    df = df.loc[:,['PatientID','Date','AdmitDate','DischargeDate','CollectionDateCdiff']].copy()
    
    # Calculate Contribution: 1 on day of positive, linear decay for 14 days afterwards.
    df['cp'] = (df.CollectionDateCdiff - df.Date).dt.days #positive prior, negative post, 0 on day of
    
    # -- Turn Missing when contribution should be 0
    df.loc[df.Date<df.CollectionDateCdiff,'cp'] = np.nan
    df.loc[df.cp < -cap,'cp'] = np.nan
#     display(df.loc[df.Cdiff==1,['PatientID','Date','CollectionDateCdiff','cp']].head(30))
#     print(df.cp.describe())

    # -- Normalize + fillna
    df['cp'] = (df['cp'] + cap)/cap # 1 on the day of, linear decay until cap
    df['cp'].fillna(0, inplace=True)
#     display(df.loc[df.Cdiff==1,['PatientID','Date','CollectionDateCdiff','cp']].head(30))

    # Hospital CP
    df['HospitalCP'] = df.groupby(['Date'])['cp'].transform('sum')
#     display(df.loc[:,['PatientID','Date','CollectionDateCdiff','cp_hospital']].sort_values('Date').head())

    # Unit CP
    df_unit = df.merge(loc[['PatientID','Date','FacilityCode','Unit']].drop_duplicates(), how='left', on=['PatientID','Date'])
    df_unit = df_unit.groupby(['Date','FacilityCode','Unit'],as_index=False)['cp'].sum()
    df_unit = loc[['PatientID','Date','FacilityCode','Unit']].drop_duplicates().merge(df_unit, how='left',on=['Date','FacilityCode','Unit'])
    df_unit = df_unit.groupby(['PatientID','Date'], as_index=False)['cp'].sum()
    df_unit.rename({'cp':'UnitCP'}, axis=1, inplace=True)
#     display(df_unit.head(30))
    df = df.merge(df_unit, how='left', on=['PatientID','Date'])
    df['UnitCP'].fillna(0, inplace=True)
#     display(df.head(30))

    df.drop(['cp','CollectionDateCdiff'],axis=1, inplace=True)
    return df

def map_feature_names(args, top_features):
    classdic = pd.read_csv(os.path.join(args['dic_data'],'MedClass_Dictionary.csv'))
    meddic = pd.read_csv(os.path.join(args['dic_data'],'Medications_Dictionary.csv'))
    meding = pd.read_csv(os.path.join(args['dic_data'],'MedIng_Dictionary.csv'))
    
    feat_names = {}
    for feat in top_features:

        if 'MedClass' in feat:
            feat_value = feat.split('_',1)[1]
            feat_names[feat] = classdic.loc[classdic.VaClassCode==feat_value,'VaClassDescription'].values[0]
        elif 'MedIng' in feat:
            feat_value = feat.split('_',1)[1]
            feat_value = float(feat_value)
            feat_names[feat] = meding.loc[meding.IngredientRxcui==feat_value,'RxnormIngredientName'].values[0]
        elif 'Medications' in feat:
            feat_value = feat.split('_',1)[1]
            feat_value = int(feat_value)
            feat_names[feat] = meddic.loc[meddic.MedicationTermID==feat_value,'MedicationName'].values[0]
    return feat_names

def process_top_features(df, feature_list):

    feat_df = df.loc[:,['PatientID','AdmitDate','DischargeDate','Date']]

    for feat in feature_list:
        case_count = 0

        for feat_type in ['LocationCode','DailyMedIng','County','DailyMedClass','DailyMedications',\
                         'DailyLabs','AdmissionType','DailyVitals','PrevMedClass','GenderCode',\
                         'PrevMedIng','PrevMedications','MaritalStatusCode','RaceName']:
            if feat_type in feat:
                feat_df[feat] = df[feat_type].str.contains(feat.split('_',1)[1])
                feat_df[feat].fillna(False, inplace=True)
                feat_df[feat] = feat_df[feat]*1
                case_count+=1

        for feat_type in ['Cdiff90','Cdiff1yr']:
            if feat_type in feat:
                feat_df[feat] = df[feat_type]
                case_count+=1

        for feat_type in ['AdmitAge','NumPrevEnc','BMI','MeanLOS','HospitalCP','UnitCP']:
            if feat_type in feat:
                low, high = feat.split('_')[1].replace('(','').replace(',','').replace(']','').split(' ')
                low, high = float(low), float(high)
                feat_df[feat] = 1*((df[feat_type]>low) & (df[feat_type]<=high))
                case_count+=1

        if case_count!=1: print('!! Number of Features Made: {}, {}'.format(case_count, feat))
        
    return feat_df

## DATA FORMAT
#####################################################################################################################
def graph_split(args,cG):
    return [cG]

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # Add self edges
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)) #.transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def format_data(args, dataset):

    cG = nx.read_gpickle(os.path.join(args['savepath'],args['dataname'],'g'+str(dataset)+'.pkl'))
    
    #Adjacency Matrix
    adj_mat_list = []
    graph_list = args['graph_split_func'](args,cG)
    for graph in graph_list:
        indices, values, shape = preprocess_adj(nx.adjacency_matrix(graph))
        adj_mat = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).to_dense()
        adj_mat_list.append(adj_mat)

    #Labels
    Y = np.array([nx.get_node_attributes(cG,'exposure')[i] for i in range(len(cG.nodes))])
    target = Variable(torch.LongTensor(Y))
    
    #Features
    S_knownpos = np.array([nx.get_node_attributes(cG,'positive')[i] for i in range(len(cG.nodes))]).reshape((-1,1))
    S_whenpos = np.array([nx.get_node_attributes(cG,'days_since_pos_norm')[i] for i in range(len(cG.nodes))]).reshape((-1,1))
    S_symp = np.array([nx.get_node_attributes(cG,'symptomatic')[i] for i in range(len(cG.nodes))]).reshape((-1,1))
    S = np.concatenate((S_knownpos,S_whenpos,S_symp), axis=1)
    S = Variable(torch.FloatTensor(S))
        
    # Pop: who should  & ID
    pop = (-1*np.array([nx.get_node_attributes(cG,'positive')[i] for i in range(len(cG.nodes))]))+1 #!pos
    gid = np.arange(len(pop)) 
        # Note: this does not have to be unique between networks b/c AUROC function takes in each network separately. 
        
    return S.cuda(), target.cuda(), [x.cuda() for x in adj_mat_list], pop, gid

## DATA LOAD ####################################################################################################################

class dataset_obj(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        
        self.args = args.copy()
        
        self.S, self.labels, self.A, self.pop = [], [], [], []
        for ind,val in enumerate(dataset):
            S_, labels_, A_, pop_, gid_= format_data(args, val)
            self.S.append(S_)
            self.labels.append(labels_)
            self.A.append(A_)
            self.pop.append(pop_)
                
        self.N = len(dataset)
        self.d = S_.shape[1]
        
    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        return self.S[idx], self.labels[idx], self.A[idx], self.pop[idx]
    
### copied from inprog146.py
def preload_data(args,datasets):
    S, target, A, pop, gid = [], [], [], [], []
    for data in datasets:
        S_, target_, A_, pop_, gid_= format_data(args, data)
        S.append(S_)
        target.append(target_)
        A.append(A_)
        pop.append(pop_)
        gid.append(gid_)
    return S, target, A, pop, gid

class data_bundler(object):
    def __init__(self, args):
        self.S_val, self.target_val, self.A_val, self.pop_val, self.gid_val = preload_data(args,np.arange(args['num_datasets_train'], args['num_datasets_train']+args['num_datasets_val']))
        
#         self.S_train, self.target_train, self.A_train, self.pop_train, self.gid_train = preload_data(args,np.arange(args['num_datasets_train']))
        
        self.S_test, self.target_test, self.A_test, self.pop_test, self.gid_test = preload_data(args,np.arange(args['num_datasets']-args['num_datasets_test'], args['num_datasets']))  
        
        self.train_data = dataset_obj(args,np.arange(args['num_datasets_train']))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args['batch_size'], shuffle=True)


## MODELS #####################################################################################################################

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 
    
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            https://github.com/LingxiaoShawn/PairNorm
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        # For some reason, Input will sometimes not have batch dim
        unsqueezed=False
        if len(x.size())<3:
            x = x.unsqueeze(0)
            unsqueezed=True
        # bs x N x out_features
        
        # Centering requires sum over N
        col_mean = x.mean(dim=1, keepdim=True)
    
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=2, keepdim=True).mean(dim=1, keepdim=True)).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        # Squeeze
        if unsqueezed:
            x = x.squeeze(0)
            
        return x
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def calc_attention(query, key, value, mask=None, dropout=None, training=None):
    "Compute 'Scaled Dot Product Attention'"
    ## query/key/value: (bs, N, d)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    ## (bs, N, N) 

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) #(mask, value): fill with value where mask is true

    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout, training)
        
    return torch.matmul(p_attn, value), p_attn

class GraphAttentionLayer(nn.Module):

    def __init__(self, args, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.args = args.copy()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.linears = clones(nn.Linear(in_features, out_features), 3)

    def forward(self, input, adj):
        ## input: (bs, N, d)
        ## adj: (bs, N, N)
        
        # For some reason, Input will sometimes not have batch dim
        unsqueezed=False
        if len(input.size())<3:
            input = input.unsqueeze(0)
            adj = adj.unsqueeze(0)
            unsqueezed=True

        # Embed Inputs
        query, key, value = [l(x) for l, x in zip(self.linears, (input, input, input))]
        N = query.size()[1]
        ## query/key/value: (bs, N, d)
        
        # Masked Attention
        h_prime, p_attn = calc_attention(query, key, value, mask=adj, dropout=self.dropout, training=self.training)
        
        if self.args['save_attn']:
            np.save(self.args['name']+'_attn.npy',p_attn.detach().cpu().numpy())
            
        # Squeeze
        if unsqueezed:
            h_prime = h_prime.squeeze(0)
            
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, args, in_features, out_features):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.args = args.copy()
        self.dropout = args['dropout']

        # GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        # - takes input & adj 
        # - returns "/sigma(AXW)", (N, out_features) (/sigma if concat=True)
        
        # Multi-headed Attention
        self.attentions = [GraphAttentionLayer(args, in_features, out_features, dropout=args['dropout'], \
                                               alpha=args['alpha'], concat=True) for _ in range(args['nheads'])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        # Put Input through Multi-head Attention 
        x = [att(x, adj) for att in self.attentions]
        ## x: (N, out_features) x nheads
        
        # Return Average
        x = torch.mean(torch.stack(x),dim=0)
        ## x: (N, out_features)
        
        return x

## LEARNING * EVALUATION #####################################################################################################################


def auroc(args,model,S,target,A,pop,gid):
    model.eval()
    output = model(S, A)
    y = target.detach().cpu().numpy()
    yhat = F.log_softmax(output, dim=1).detach().cpu().numpy()
    
    # Patient Level Max, No Locations
    df = pd.DataFrame({'y':y, 'yhat':yhat[:,1].flatten(), 'pop':pop, 'gid':gid})
    df = df.groupby(['gid'], as_index=False).max()
    df = df.loc[df['pop']==1,:]
            
    return df.yhat.values, df.y.values


def train_models(args, dbundle):
    acc = {'val':[], 'test':[]}
    trainval_loss = np.zeros((args['budget'],args['epochs'],2))
    args_list = []
    args['d'] = dbundle.train_data.d

    # Train
    best_loss_over_runs = 10**10
    best_auc_over_runs = 0
    for budget in range(args['budget']):

        print('-- Budget {}'.format(budget))
        
        #HP 
        for key in args['HP'].keys():
            args[key]= np.random.choice(args['HP'][key])
        args_list.append(args.copy())
        
        model = args['GCN'](args).cuda()
        best_loss = 10**10
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args['l2'])

        #Train
        patience = args['patience']
        lr=0.001 #default ADAM LR
        for epoch in range(args['epochs']):

            # Train
            loss_train = []
            model.train()
            for batch_idx, (S, target, A, pop) in enumerate(dbundle.train_loader):
                optimizer.zero_grad() 
                output = model(S, A)
                
                pop_ind = np.where(pop.flatten()==1)[0]
                output = output[:,pop_ind,:]
                target = target[:,pop_ind]
                
                loss = F.cross_entropy(output.view(-1,2),target.view(-1), weight=args['weight'].cuda())
                loss.backward()
                optimizer.step()
                loss_train.append(loss.cpu().detach().numpy())
            loss_train = np.mean(loss_train)
            
            # Val
            loss_val = []
            for S_, target_, A_, pop_ in zip(dbundle.S_val, dbundle.target_val, dbundle.A_val, dbundle.pop_val):
                output_ = model(S_, A_)
                
                pop_ind_ = np.where(pop_==1)[0]
                output_ = output_[pop_ind_,:]
                target_ = target_[pop_ind_]
                
                loss_ = F.cross_entropy(output_, target_, weight=args['weight'].cuda())
                loss_val.append(loss_.cpu().detach().numpy())
            loss_val = np.mean(loss_val)
            trainval_loss[budget,epoch,:]=[loss_train, loss_val]

            if (epoch%args['save_epoch']==0)|(epoch==args['epochs']-1): print('-- -- Epoch {}: loss {:.3f}, {:.3f}'.format(epoch,loss_train,loss_val))

            # Save Best Model
            if loss_val<best_loss:
                torch.save(model.state_dict(), args['name']+'checkpoint.pth.tar')
                best_loss = loss_val
                patience = args['patience']
            else:
                patience = patience - 1
                if patience==0:
                    # Take a Step
                    if lr>.0001:
                        lr = lr*.1 #similar to taking a step via StepLR
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = args['patience']
                    else:
                        break

        #Load Best Model
        model = args['GCN'](args).cuda()
        model.load_state_dict(torch.load(args['name']+'checkpoint.pth.tar'))
        model.eval()

        #Evaluate
        yhat_y = {'yval':[], 'yhatval':[], 'ytest':[], 'yhattest':[], 'ytrain':[], 'yhattrain':[]}
        # -- Train/ Val/ Test
        for S__, target__, A__, pop__, gid__, i in [\
                                                    (dbundle.S_val, dbundle.target_val, dbundle.A_val,dbundle.pop_val,dbundle.gid_val, 'val'),\
                                                    (dbundle.S_test, dbundle.target_test, dbundle.A_test,dbundle.pop_test,dbundle.gid_test, 'test')]:
            # -- For each network in Train/Val/Test
            for S_, target_, A_, pop_, gid_ in zip(S__, target__, A__, pop__, gid__):
                
                yhat_, y_= auroc(args, model, S_, target_, A_, pop_, gid_)
                yhat_y['y'+i].append(y_)
                yhat_y['yhat'+i].append(yhat_)
                
            # -- Calculate AUROC to match LogReg's Calculation (Over all datasets)
            yhat_y['yhat'+i] = np.concatenate(yhat_y['yhat'+i], axis=0)
            yhat_y['y'+i] = np.concatenate(yhat_y['y'+i], axis=0)
            acc[i].append(roc_auc_score(yhat_y['y'+i], yhat_y['yhat'+i]))
                #Note: this assumes we go in order train-> val-> test
            

        #Save Best Model Over Runs 
        if acc['val'][budget] > best_auc_over_runs:
            # Note: this is 0 b/c we want val
            torch.save(model.state_dict(), args['name']+'.pth.tar')
            best_auc_over_runs = acc['val'][budget]
            np.save(os.path.join(args['savepath'],args['name']+"args.npy"),args)
            
            
    os.remove(args['name']+'checkpoint.pth.tar')
    np.savez(os.path.join(args['savepath'],args['name']), acc=acc, trainval_loss=trainval_loss, args_list=args_list)
    return 


#######################################################################################################################
## MODEL SPECIFIC #####################################################################################################
#######################################################################################################################

class GCNa11(nn.Module):

    def __init__(self, args):
        super(GCNa11, self).__init__()
        
        self.args = args.copy()
        ''' KEYS
        * residual: T/F, add residual connections or not
        * ego_v_nbr: T/F, separate ego vs neighbor representation
        * kdeg_nbr: T/F, incorporates kth degree neighbors separately
        * attention: T/F, GAT vs GCN
            * alpha, nheads
        
        * hidden_size
        * depth
        * d: input size
        * output_size
        '''
        
        if self.args['kdeg_nbr']:
            ego_multiplier = 3
            normal_multiplier = 2
        else:
            ego_multiplier = 2
            normal_multiplier = 1
        
        if self.args['ego_v_nbr']:
            multiplier = ego_multiplier 
        else: 
            multiplier = normal_multiplier
        
        if self.args['residual']:
            fc_input_size = self.args['hidden_size']*self.args['depth']+self.args['d']
        else: 
            fc_input_size = self.args['hidden_size']
            
        # Elements
        self.fc = nn.Linear(fc_input_size,args['output_size'])
        self.norm = PairNorm()
        
        # -- GAT
        if self.args['attention']:
            if self.args['ego_v_nbr']:
                raise ValueError('Have not implemented attention w. ego_v_nbr functionality')
            if self.args['kdeg_nbr']:
                raise ValueError('Have not implemented attention w. kdeg_nbr functionality')
            self.gc_layers = nn.ModuleList([
                GAT(args, in_features=(self.args['d']*multiplier if layer==0 else self.args['hidden_size']*multiplier),
                                    out_features=(self.args['hidden_size']))
                for layer in range(self.args['depth'])])
        # -- GCN
        else:
            self.gc_layers = nn.ModuleList([
                nn.Linear(in_features=(self.args['d']*multiplier if layer==0 else self.args['hidden_size']*multiplier),
                                out_features=(self.args['hidden_size']))
                for layer in range(self.args['depth'])
            ])

        

        
    def forward(self, S, A, kA=None):  
        embedding = S
        A = A[0]
        
        if (self.args['residual']) | (self.args['ego_v_nbr']):
            embedding_old_list = [S]
        
        for layer in range(self.args['depth']):
            
            # GAT
            if self.args['attention']:
                embedding = self.gc_layers[layer](embedding,A)
            
            # GCN
            else:
                input_list = [torch.matmul(A, embedding)]
                if self.args['kdeg_nbr']:
                    if not kA:
                        # use 2 degree kA;
                        kA = A@A
                        # normalize; D**(-0.5) A D**(-0.5); shape(kA) = (bs, n, n)
                        d_inv_sqrt = kA.sum(-1) ** (-0.5) # row sum
                        # batch version of diag
                        d_inv_sqrt = torch.diag_embed(d_inv_sqrt)
                        kA = d_inv_sqrt @ kA @ d_inv_sqrt
                        
                    input_list.append(torch.matmul(kA, embedding))
                if self.args['ego_v_nbr']:
                    input_list.append(embedding_old_list[-1])
                embedding = self.gc_layers[layer](torch.cat(input_list, dim=-1))
                
            embedding = F.relu(self.norm(embedding))
            
            if (self.args['residual']) | (self.args['ego_v_nbr']):
                embedding_old_list.append(embedding)
                
            if layer==(self.args['depth']-1):
                if self.args['residual']:
                    output = F.relu(self.fc(torch.cat(embedding_old_list,dim=-1)))
                else:
                    output = F.relu(self.fc(embedding))
                    
            else:    
                embedding = F.dropout(embedding, self.args['dropout'], training=self.training)
        return output  
