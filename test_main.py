import numpy as np
import pandas as pd
import argparse
import os
import re
import pickle
import sys
import datetime
from shutil import copyfile
import torch
import scipy
import pdb

import test_data as custom_data
import test_function as custom_function
import test_model as custom_model

# Author: Jeeheh Oh
# Date Created: 2019.05.31
# Purpose: This file contains all the settings. This is the only file that should need to be run.

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Set/Prompt for Experiment Settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
save_code_list = ['test_main.py','test_data.py','test_function.py','test_model.py']

parser=argparse.ArgumentParser()

parser.add_argument('--id', type=str,help='Determines save name')
parser.add_argument('--model',type=str,help='Determines which model is used')
parser.add_argument('--cuda',type=int)
parser.add_argument('--task',type=str) #cdi, neighbors, neighbor_deciles

parser.add_argument('--tempjob',type=str) 

#Data
parser.add_argument('--data_loc',type=str,default='/data4/jeeheh/ShadowPeriod/Data/UM_CDI/',help='Location of data')
parser.add_argument('--locdata_loc',type=str,default='/data4/jeeheh/ShadowPeriod/Toy/Toy_locations.csv')
parser.add_argument('--auxdata_loc',type=str,default='/data4/jeeheh/ShadowPeriod/Toy/Toy_CDI.csv')
# parser.add_argument('--INSTANCE_KEYS',nargs='+',type=str,default=['season_title_id', 'country_iso_code', 'days_since_launch'])
# parser.add_argument('--sig_T',type=int,default=14)

# #Phantom Step: Adds in t=-91 for PostGL
# parser.add_argument('--phantom_step', dest='phantom_step', action='store_true')
# parser.add_argument('--no_phantom_step', dest='phantom_step', action='store_false')
# parser.set_defaults(phantom_step=True)

#Hyperparameters
parser.add_argument('--epochs',type=int,default=20) 
parser.add_argument('--budget',type=int,default=5) 

args=parser.parse_args()
args=vars(args)

#Prompt for missing args
for x in args.keys():
    if args[x] is None:
        args[x]=input("What is the {} of this experiment? ".format(x))
        print('--{}: {}'.format(str(x),str(args[x])))
    else: 
        print("--{}: {}".format(str(x),str(args[x])))
        
#Cuda
args['cuda']=int(args['cuda'])
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))
    torch.set_num_threads(4)
    
#Task
if args['task'] == 'cdi': 
    args['classification'] = True
    args['num_classes'] = 2
elif args['task'] in ['neighbors','unique_neighbors']:
    args['classification'] = False
    args['num_classes'] = 1
elif args['task'] == 'neighbor_deciles':
    args['classification'] = True
    args['num_classes'] = 10
else:
    raise ValueError('Error: No task')

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Save Settings/ Create experiment output folder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Create Subfolders for Experiment Results
# >> Folder created for each model
# >> Folder created for each experiment id within model.

# SUBFOLDER = re.sub('[^\w]', '_', args['model']).lower()
SUBFOLDER = args['model']
try:
    os.makedirs(SUBFOLDER)
except:
    pass

try:
    os.makedirs(os.path.join(SUBFOLDER,args['id']))
except:
    continue_input = input('This folder ({}) exists. You will overwrite existing experiment data. Continue? (yes/no): '.format(args['model']+"/"+args['id']))
    if continue_input=='yes':
        pass
    else:
        sys.exit()
        
args['save_folder']=SUBFOLDER

#Copy versions of code
for f in save_code_list:
    copyfile(f, os.path.join(SUBFOLDER,args['id'],f))
    # This will overwrite existing code. 
args['save_code_list'] = save_code_list

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Format Data and Save

trainx, trainy, trainkey, valx, valy, valkey, testx, testy, testkey, loc_df, cdi_df = custom_data.format_data(args)

trainkey.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'train')
valkey.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'val')
testkey.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'test')
cdi_df.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'cdi_df')
loc_df.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'loc_df')

for filename, var in [('trainx',trainx), ('valx',valx), ('testx',testx)]:
    scipy.sparse.save_npz(os.path.join(args['save_folder'],args['id'],'data'+filename+'_temp.npz'), var)

for filename, var in [('trainy',trainy), ('valy',valy), ('testy',testy)]:
    np.save(os.path.join(args['save_folder'],args['id'],'data'+filename+'_temp.npy'),var)


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Models and Experiments
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Save copy of args in Subfolder        
with open(os.path.join(args['save_folder'],args['id'],'args.pkl'), 'wb') as f:
    pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    # This will overwrite existing text. This does not append.
    
if args['model']=='pytorchLSTM':
    custom_function.HPsearch(args,custom_model.pytorchLSTM)
    
if args['model']=='transformer':
    custom_function.HPsearch(args,custom_model.transformer)
    
if args['model']=='gcn':
    custom_function.HPsearch(args,custom_model.gcn_lstm)

if args['model']=='simple_gcn':
    custom_function.HPsearch(args,custom_model.simple_gcn_lstm)

#Save copy of args in Subfolder        
with open(os.path.join(args['save_folder'],args['id'],'args.pkl'), 'wb') as f:
    pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    # This will overwrite existing text. This does not append.
    
#Delete Data files
for filename in ['datakeys_temp.h5','datatrainx_temp.npz','datatrainy_temp.npy'
                 ,'datatestx_temp.npz','datatesty_temp.npy'
                ,'datavalx_temp.npz','datavaly_temp.npy']:
    os.remove(os.path.join(args['save_folder'],args['id'],filename))