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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import test_data as custom_data
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
                          'depth','dropout','heads','hidden_size_gcn','embedding_size']
        args['randominit'] = int(randominit)
        args['hidden_size_gcn'] = int(np.random.choice([50, 100, 200, 400]))
        args['learning_rate'] = np.random.choice([0.1, 0.01])
        # args['dropout'] = np.random.choice([.05,.1,.15,.2,.25]) #RNN dropout currently hardcoded.
        args['dropout'] = np.random.choice([0.1, 0.2, 0.4, 0.6])
        args['heads'] = int(np.random.choice([3,1,19])) #The choice of heads depends on args['d'], heads should be a divisor of d.
        args['embedding_size'] = int(np.random.choice([50, 75, 100, 150]))

        
        if args['model']=='transformer':
            args['batch_size'] = int(np.random.choice([25, 50, 75, 100])) 
            args['l2'] = np.random.choice([0.001, 0.005, 0.01]) 
            args['hidden_size'] = int(np.random.choice([100,150,200,250,300])) 
            args['depth'] = int(np.random.choice([2,3,4,5])) 

        elif args['model']=='pytorchLSTM':
            args['batch_size'] = int(np.random.choice([10, 25, 50, 75]))
            # args['l2'] = np.random.choice([0.001, 0.01, 0.1])
            args['l2'] = np.random.choice([0.01, 0.05, 0.1, 0.15])
            args['depth'] = int(np.random.choice([1,2]))
            if args['depth']==1:
                args['hidden_size'] = int(np.random.choice([600,900,1200,1500]))
            elif args['depth']==2:
                args['hidden_size'] = int(np.random.choice([450,600]))
        
        elif args['model']=='gcn':
            args['batch_size'] = int(np.random.choice([10, 25, 50, 75]))
            args['l2'] = np.random.choice([0.01, 0.05, 0.1, 0.2, 0.4])
            args['depth'] = int(np.random.choice([1,2]))
            if args['depth']==1:
                args['hidden_size'] = int(np.random.choice([600,900,1200]))
            elif args['depth']==2:
                args['hidden_size'] = int(np.random.choice([450,600]))


        #Learn Model
        print('hidden size {}, depth {}'.format(args['hidden_size'], args['depth']))
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
    print('num params: {}'.format(count_parameters(model)))
    
    #optimizer = torch.optim.SGD(model.parameters(),lr=args['learning_rate'], weight_decay=args['l2']) 
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args['l2'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    #Epochs
#     earlystop = []
    patience=11
    best_val_auc = None
    for epoch in range(1, args['epochs']+1):
        print('epoch: {}'.format(epoch))
#         print('max memory cached {}'.format(torch.cuda.max_memory_cached(device=args['cuda'])))
#         print('max memory allocated {}'.format(torch.cuda.max_memory_allocated(device=args['cuda'])))
#         torch.cuda.reset_max_memory_cached(device=args['cuda'])
#         torch.cuda.reset_max_memory_allocated(device=args['cuda'])
            
        args['current_epoch'] = epoch

        #Train
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
            patience = 10
        else: patience = patience -1 

        #Early Stopping
        if patience==0:
            break

        scheduler.step()
            
    #Prepare Pred DF    
    pred_df = pred_df.append(best_zdf)
    pred_df = pred_df.append(best_val_zdf)
    
    pred_df['num_param']=[count_parameters(model)]*pred_df.shape[0]
    
    for feature in HP_feature_list: 
        pred_df[feature]=[args[feature]]*pred_df.shape[0]
    
    return pred_df, traintestloss_df



def train(train_loader, model, args, optimizer):
    model.train()
    for batch_idx, (data, target, seqlen, indices, A, S) in enumerate(train_loader):
        
        data, target, S, A, seqlen = Variable(data), Variable(target), Variable(S), Variable(A), Variable(seqlen)
        optimizer.zero_grad()  
        output = model(data, S, A, seqlen)
                
        loss, _ = loss_opt(args,output,target,seqlen)
        loss.backward()
        optimizer.step()
            
    return

# Target Replication
# def loss_opt(args, output, target, seqlen):
#     output = output.view(-1,args['num_classes'])
#     target = target.flatten()

#     maxseqlen = max(seqlen)
#     mask = torch.arange(maxseqlen)[None, :]< torch.FloatTensor(seqlen)[:, None]
#     mask = mask.type(torch.FloatTensor).flatten()        
#     loss = torch.sum(F.cross_entropy(output,target,reduction='none')*mask.cuda())
#     return loss, sum(seqlen)

# Subsample 3
def loss_opt(args, output, target, seqlen):
    output = output.view(-1,args['num_classes'])
    target = target.flatten()
    
    seqlen = seqlen.cpu().numpy().flatten().astype('int')
    maxseqlen = max(seqlen)    
    subsmpl = [np.random.choice(x, size=3, replace=True) for x in seqlen]
    # replace=True because of the case where seqlen = 2. This can happen if day of CDI is 4 in the training set.
    
    mask1 = torch.arange(maxseqlen)[None,:] == torch.FloatTensor([x[0] for x in subsmpl])[:,None]
    mask2 = torch.arange(maxseqlen)[None,:] == torch.FloatTensor([x[1] for x in subsmpl])[:,None]
    mask3 = torch.arange(maxseqlen)[None,:] == torch.FloatTensor([x[2] for x in subsmpl])[:,None]
    
    mask = torch.max(mask1.type(torch.FloatTensor),mask2.type(torch.FloatTensor))
    mask = torch.max(mask, mask3.type(torch.FloatTensor)).flatten()
    
    loss = torch.sum(F.cross_entropy(output,target,reduction='none')*mask.cuda())
    return loss, 3*len(seqlen) #this is an approximation because there will be cases where it should be 2 and not 3.

# Max Onwards
# def loss_opt(args, output, target, seqlen):
    
#     # Padding Mask
#     maxseqlen = max(seqlen)
#     mask = torch.arange(maxseqlen)[None, :]< torch.FloatTensor(seqlen)[:, None]
#     mask = mask.type(torch.FloatTensor)
#     #print(mask.shape) # batch_size x maxseqlen

#     # Max Onwards Mask
#     #print(output.shape) # batch_size x maxseqlen x num_classes
# #     with torch.no_grad():
# #         _, argmax = torch.max(F.log_softmax(output,dim=2)[:,:,1]*mask.cuda(), dim=1)
# #         argmax = argmax.cpu().numpy()
#         #Hard coding ASSUMES classification task
    
#     _, argmax = torch.max(F.log_softmax(output.detach(),dim=2)[:,:,1]*mask.cuda(), dim=1)
#     argmax = argmax.cpu().numpy()    
#     maxmask = torch.arange(maxseqlen)[None, :]>= torch.FloatTensor(argmax)[:, None]
        
#     # Combine Masks
#     mask = torch.min(mask, maxmask.type(torch.FloatTensor)).flatten()
#     output = output.view(-1,args['num_classes'])
#     target = target.flatten()
    
#     loss = torch.sum(F.cross_entropy(output,target,reduction='none')*mask.cuda())
#     return loss, sum(mask).numpy()

def test(test_loader, model, args, dataset):
    
    model.eval()    
    with torch.no_grad():

        total_loss=0
        total_sum=0
        df=pd.DataFrame()

        for data, labels, seqlen, indices, A, S in test_loader:

            data, labels, S, A, seqlen = Variable(data), Variable(labels), Variable(S), Variable(A), Variable(seqlen)

            #Calculate Loss 
            output = model(data, S, A, seqlen)
            loss, denom_add = loss_opt(args, output, labels, seqlen)
            total_loss = total_loss + loss.detach().cpu().numpy()
            total_sum = total_sum + denom_add

            #Calculate AUROC (Evaluation Metric: taking the max)
            outputs = F.softmax(model(data, S, A, seqlen),dim=2)[:,:,1]
            # likelihood of CDI positive, ranges from 0-1
            # batch_size x max_seqlen x num_classes

            #Create Mask
            maxseqlen = max(seqlen)
            mask = torch.arange(maxseqlen)[None, :]< seqlen[:, None].cpu()
            mask = mask.type(torch.FloatTensor)

            #Take the max per admission with mask
            outputs = torch.max(outputs*mask.cuda(),dim=1).values
            labels = torch.max(labels,dim=1).values

            df = df.append(pd.DataFrame({'labels':labels.cpu().numpy(),
                                        'outputs':outputs.cpu().numpy(),
                                        'eid':test_loader.dataset.datakey.loc[indices,'eid']}))

        auroc = roc_auc_score(df.labels.values, df.outputs.values)
        print('--{} auc: {:.5f}'.format(dataset,auroc))
        print('--{} celoss: {:.5f}'.format(dataset,total_loss/total_sum))
        df['dataset'] = [dataset]*df.shape[0]

    return df, auroc, total_loss/total_sum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_grad_flow(named_parameters, args):
    #https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(os.path.join(args['save_folder'],args['id'],'rinit'+str(args['randominit'])+'ep'+str(args['current_epoch'])+'.png'))