import numpy as np
import pandas as pd
import argparse
import os
import re
import pickle
import sys
import datetime
from shutil import copyfile
from sklearn.linear_model import LinearRegression
import torch
import scipy
import pdb

import data as custom_data
import function as custom_function
import model as custom_model

# Author: Jeeheh Oh
# Date Created: 2019.05.31
# Purpose: This file contains all the settings. This is the only file that should need to be run.

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Set/Prompt for Experiment Settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
save_code_list = ['main.py','data.py','function.py','model.py']

parser=argparse.ArgumentParser()

parser.add_argument('--id', type=str, default='NEED',help='Determines save name')
parser.add_argument('--model',type=str, default='NEED',help='Determines which model is used')

#Data
parser.add_argument('--data_loc',type=str,default='/data4/jeeheh/ShadowPeriod/Data/Adult_fulldata_2_14_19_v16 - labs and meds fixed - Without Procedures BMI Fixed/',help='Location of data')
parser.add_argument('--INSTANCE_KEYS',nargs='+',type=str,default=['season_title_id', 'country_iso_code', 'days_since_launch'])
parser.add_argument('--num_classes',type=int,default=1)
parser.add_argument('--time_min',type=int,default=-91,help='Determines data time range')
parser.add_argument('--time_max',type=int,default=27,help='Determines data time range')
parser.add_argument('--time_step',type=int,default=5,help='Determines which time steps are used to determine early stopping criteria')

#Phantom Step: Adds in t=-91 for PostGL
parser.add_argument('--phantom_step', dest='phantom_step', action='store_true')
parser.add_argument('--no_phantom_step', dest='phantom_step', action='store_false')
parser.set_defaults(phantom_step=True)

#Hyperparameters
parser.add_argument('--epochs',type=int,default=30) 
parser.add_argument('--budget',type=int,default=30) 

args=parser.parse_args()
args=vars(args)

#Check for GPUs
if torch.cuda.device_count()>0:
    args['use_cuda']=True
else: args['use_cuda']=False

#Prompt for missing args
for x in args.keys():
    if (args[x]=='NEED'):
        args[x]=input("What is the {} of this experiment? ".format(x))
        print('--{}: {}'.format(str(x),str(args[x])))
    else: 
        print("--{}: {}".format(str(x),str(args[x])))
        
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
trainx, trainy, trainkey, testx, testy, testkey = custom_data.format_data(args)
trainkey.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'train')
testkey.to_hdf(os.path.join(args['save_folder'],args['id'],'datakeys_temp.h5'),'test')

for filename, var in [('trainx',trainx), ('testx',testx)]:
    scipy.sparse.save_npz(os.path.join(args['save_folder'],args['id'],'data'+filename+'_temp.npz'), var)
    
for filename, var in [('trainy',trainy),('testy',testy)]:
    np.save(os.path.join(args['save_folder'],args['id'],'data'+filename+'_temp.npy'),var)


#### >>> need to unpack? evaluation...
    
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
    
#Save copy of args in Subfolder        
with open(os.path.join(args['save_folder'],args['id'],'args.pkl'), 'wb') as f:
    pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    # This will overwrite existing text. This does not append.
    
#Delete Data files
# for filename in ['datakeys_temp.h5','datatrainx_temp.npz','datatrainy_temp.npy'
#                  ,'datatestx_temp.npz','datatesty_temp.npy']:
#     os.remove(os.path.join(args['save_folder'],args['id'],filename))