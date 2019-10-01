import numpy as np
import pandas as pd
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
import pickle
import pdb

import data as custom_data
from sklearn.metrics import roc_auc_score


#############################################################################################################

def HPsearch(args,Net):

    #Init Empty
    pred_df=pd.DataFrame({})
    traintestloss_df=pd.DataFrame({})

    #Random Init/Hyperparameter Search Budget
    for randominit in range(args['budget']):
        print('budget {}/{}'.format(randominit+1,args['budget']))
        
        #Set HP
        HP_feature_list = ['batch_size','learning_rate','l2','hidden_size','randominit',
                          'depth','dropout','heads']
        args['randominit'] = int(randominit)
        
        args['learning_rate'] = np.random.choice([0.1, 0.01])
        args['dropout'] = np.random.choice([.05,.1,.15,.2,.25]) #RNN dropout currently hardcoded.
        args['heads'] = int(np.random.choice([3,1,19])) #The choice of heads depends on args['d'], heads should be a divisor of d.
        
        if args['model']=='transformer':
            args['batch_size'] = int(np.random.choice([25, 50, 75, 100])) 
            args['l2'] = np.random.choice([0.001, 0.005, 0.01]) 
            args['hidden_size'] = int(np.random.choice([100,150,200,250,300])) 
            args['depth'] = int(np.random.choice([2,3,4,5])) 

        elif args['model']=='pytorchLSTM':
            args['batch_size'] = int(np.random.choice([10, 25, 50, 75]))
            args['l2'] = np.random.choice([0.001, 0.01, 0.1])
            args['hidden_size'] = int(np.random.choice([100,300,600,900])) 
            args['depth'] = 1
                    
        #Learn Model
        zpred_df, ztraintestloss_df = learn_model(args,Net,HP_feature_list)

        #Update DF
        traintestloss_df = traintestloss_df.append(ztraintestloss_df, ignore_index=True)
        pred_df = pred_df.append(zpred_df, ignore_index=True)
                
        #Save DF
        pred_df.to_hdf(os.path.join(args['save_folder'],args['id'],'data.h5'),'pred')
        traintestloss_df.to_hdf(os.path.join(args['save_folder'],args['id'],'data.h5'),'traintestloss')
    return 
    

def learn_model(args,Net,HP_feature_list):
    
    #Init
    pred_df=pd.DataFrame({})
    traintestloss_df=pd.DataFrame({})
    
    #Load Data
    train_data = custom_data.dataset_obj(args,'train')
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True, 
                                               collate_fn=custom_data.custom_collate_fn)
    val_data = custom_data.dataset_obj(args,'val')
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['batch_size'],shuffle=False, 
                                               collate_fn=custom_data.custom_collate_fn)
    test_data = custom_data.dataset_obj(args,'test')
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=args['batch_size'],shuffle=False,
                                             collate_fn=custom_data.custom_collate_fn)
    
    args['d']=int(train_data.d)
    print('-- load data finished')

    #Init Model
    model = Net(args)
    if args['use_cuda']:
        model.cuda()

    #Epochs
    earlystop = []
    best_val_auc = None
    for epoch in range(1, args['epochs']+1):
        print('epoch: {}'.format(epoch))

        #Train
        #optimizer = torch.optim.SGD(model.parameters(),lr=args['learning_rate'], weight_decay=args['l2']) 
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args['l2'])
        train(train_loader, model, args, optimizer)

        #Evaluate
        zdf, testauc, testloss = test(test_loader, model, args, 'test')  
        val_zdf, valauc, valloss = test(val_loader, model, args, 'val')
        _, trainauc, trainloss = test(train_loader, model, args, 'train')

        #Train/Val/Test Loss
        traintestloss_df = traintestloss_df.append(pd.DataFrame({'testauc':[testauc], 
                                                                 'valauc':[valauc], 
                                                                 'trainauc':[trainauc],
                                                                 'testloss':[testloss],
                                                                 'valloss':[valloss],
                                                                 'trainloss':[trainloss],
                                                                 'epoch':[epoch],
                                                                 'randominit':[args['randominit']]}))

        #Val Loss
        if best_val_auc is None: best_val_auc = valauc
        elif valauc > best_val_auc: best_val_auc = valauc

        #Update Output/Target dicts & Save Models
        if valauc==best_val_auc:
            best_zdf = zdf
            best_val_zdf = val_zdf
            torch.save(model.state_dict(), os.path.join(args['save_folder'],
                                        args['id'],'model'+str(args['randominit'])+'.pth'))

        #Early Stopping
        if len(earlystop)>5:
            if abs(earlystop.pop(0)-valauc)<.0001:
                break
        earlystop.append(valauc)
            
    #Prepare Pred DF    
    pred_df = pred_df.append(best_zdf)
    pred_df = pred_df.append(best_val_zdf)
    
    pred_df['num_param']=[count_parameters(model)]*pred_df.shape[0]
    
    for feature in HP_feature_list: 
        pred_df[feature]=[args[feature]]*pred_df.shape[0]
    
    return pred_df, traintestloss_df



def train(train_loader, model, args, optimizer):
    model.train()
    for batch_idx, (data, target, seqlen, indices) in enumerate(train_loader):
        
#         if args['use_cuda']:
#             data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  
        output = model(data)
                
        output = output.view(-1,args['num_classes'])
        target = target.flatten()
                             
        #Create Mask
        maxseqlen = max(seqlen)
        mask = torch.arange(maxseqlen)[None, :]< torch.FloatTensor(seqlen)[:, None]
        mask = mask.type(torch.FloatTensor).flatten()
        
        loss = torch.sum(F.cross_entropy(output,target,reduction='none')*mask.cuda())
        loss.backward()
        optimizer.step()
            
    return



def test(test_loader, model, args, dataset):
    
    model.eval()    
    
#     alllabels=np.empty(0)
#     alloutputs=np.empty(0)
    total_loss=0
    total_sum=0
    df=pd.DataFrame()

    for data, labels, seqlen, indices in test_loader:

#         if args['use_cuda']:
#             data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        
        #Calculate Loss (Target Replication Loss)
        output = model(data)
        output = output.view(-1,args['num_classes'])
        target = labels.flatten()
                             
        maxseqlen = max(seqlen)
        mask = torch.arange(maxseqlen)[None, :]< torch.FloatTensor(seqlen)[:, None]
        mask = mask.type(torch.FloatTensor).flatten()        
        loss = torch.sum(F.cross_entropy(output,target,reduction='none')*mask.cuda())
        total_loss = total_loss + loss.detach().cpu().numpy()
        total_sum = total_sum + sum(seqlen)

        #Calculate AUROC (Evaluation Metric: taking the max)
        outputs = F.softmax(model(data),dim=2)[:,:,1]
        # likelihood of CDI positive, ranges from 0-1
        # batch_size x max_seqlen x num_classes

        #Create Mask
        maxseqlen = max(seqlen)
        mask = torch.arange(maxseqlen)[None, :]< torch.FloatTensor(seqlen)[:, None]
        mask = mask.type(torch.FloatTensor)

        #Take the max per admission with mask
        with torch.no_grad():
            outputs = torch.max(outputs*mask.cuda(),dim=1).values
            labels = torch.max(labels,dim=1).values

        df = df.append(pd.DataFrame({'labels':labels.cpu().numpy(),
                                    'outputs':outputs.cpu().numpy(),
                                    'eid':test_loader.dataset.datakey.loc[indices,'eid']}))
#         if alllabels.shape[0]==0:
#             alllabels = labels.cpu().numpy()
#             alloutputs = outputs.cpu().numpy()
#         else:
#             alllabels = np.hstack([alllabels,labels.cpu().numpy()])
#             alloutputs = np.hstack([alloutputs,outputs.cpu().numpy()])
            
#     auroc = roc_auc_score(alllabels, alloutputs)
    auroc = roc_auc_score(df.labels.values, df.outputs.values)
    print('--{} auc: {:.5f}'.format(dataset,auroc))
    print('--{} celoss: {:.5f}'.format(dataset,total_loss/total_sum))
    df['dataset'] = [dataset]*df.shape[0]

    return df, auroc, total_loss/total_sum

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

