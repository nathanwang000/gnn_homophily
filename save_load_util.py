"""
Contains functions for saving and loading the data

Created by Ryen Krusinga
"""
import numpy as np
import os
import pickle
import scipy.sparse
import datetime as dt

# fname: a file name to save the array to
# arr: a sparse matrix of type scipy.sparse.csr_matrix
# Save arr to file
def save_sparse_csr(fname, arr):
	np.savez(fname,data=arr.data,indices=arr.indices,indptr=arr.indptr,shape=arr.shape)

# fname: the file to load
# Loads a scipy.sparse.csr_matrix from a file
def load_sparse_csr(fname):
	loader = np.load(fname)
	return scipy.sparse.csr_matrix((loader['data'],loader['indices'],loader['indptr']),shape=loader['shape'])

# trainfile: file containing xtrain
# testfile: file containing xtest
# referencefile: file containing reference variable columns like PID and EncID
# pklfile: file containing other relevant data like column names and a feature histogram
def load_all(trainfile, testfile, referencefile, pklfile, path):
	xtrain = load_sparse_csr(os.path.join(path,trainfile))
	xtest = load_sparse_csr(os.path.join(path,testfile))
# 	refloader = np.load(os.path.join(path,referencefile))
	refloader = np.load(os.path.join(path,referencefile),allow_pickle=True)
	ytrain = refloader['ytrain']
	ytest = refloader['ytest']
	# Reference variables (PID, EncID, Daynums, etc.)
	trainref = refloader['trainref']
	testref = refloader['testref']

	# Dictionaries mapping column names to indices
	cols_hist = pickle.load(open(os.path.join(path,pklfile),'rb'))
	ref_cols = cols_hist[0]
	feat_cols = cols_hist[1]
	label_cols = cols_hist[2]
	feat_hist = cols_hist[3]
	# Histogram of how many binary features each
	# original column was expanded to

	return xtrain, ytrain, xtest, ytest, trainref, testref, ref_cols, feat_cols, label_cols, feat_hist

def load_simulated(n, d, randstate = 1):
	np.random.seed(randstate)
	xtrain = np.random.randint(0,2,(n,d))
	xtest = np.random.randint(0,2,(n,d))
	ytrain = np.random.randint(0,2,(n))
	ytest = np.random.randint(0,2,(n))
	trainref = np.random.randint(0,n/4,(n,3))
	testref = np.random.randint(n/4,n/2,(n,3))
	ref_cols = {'PatientID':0,'EncounterID':1,'Day':2}
	feat_cols = {('col'+str(i)):i for i in np.arange(n)}
	label_cols = {'label':0}
	feat_hist = {}

	return scipy.sparse.csr_matrix(xtrain), ytrain, scipy.sparse.csr_matrix(xtest), ytest, trainref, testref, ref_cols, feat_cols, label_cols, feat_hist


# col should be a list of columns to remove
# feature is a set of feature vectors
# Want to remove DailyLabs_CDIFF_N I think
# Readjusts numbers to fit
def check_ref_cols(rc):
	if rc[0,1] != 0:
		return False
	for i in range(1,len(rc)):
		if rc[i,1]-rc[i-1,1] != 1:
			return False
	return True

	
# Get a list of all columns *starting* with a substring
def get_cols_startingwith(refcols, substr):
	return [v for v in refcols if v.startswith(substr)]

# Get a list of all columns with the substring
def get_cols_with(refcols, substr):
	return [v for v in refcols if substr in v]

def remove_cols(feat, refcols, col):
	newrefcols = np.empty((len(refcols),2),dtype=object)
	for i, key in enumerate(refcols):
		newrefcols[i,0] = key
		newrefcols[i,1] = refcols[key]
	newrefcols = newrefcols[newrefcols[:,1].argsort()]
	# Definitely rename these variables

	assert(check_ref_cols(newrefcols))
	
	col_inds = np.ones((feat.shape[1]),dtype=bool)
	for c in col:
		ind = refcols[c]
		for i in range(7): # Because of the repeats
			col_inds[ind+i*len(refcols)//7] = False
	
	new_feats = feat[:,col_inds]
	nref = newrefcols[col_inds[:len(col_inds)//7],0]
	ref = {v:i for i,v in enumerate(nref)}

	return new_feats, ref

# year should be a string, like '2014'
def load_year(year, path):
	trainref = np.load(os.path.join(path,'trainref_fixed'+year+'.npy'))
	xtrain = load_sparse_csr(os.path.join(path,'xtrain_fixed'+year+'.npz'))
	ytrain = np.load(os.path.join(path,'ytrain_fixed'+year+'.npy'))
	return trainref, xtrain, ytrain

def extract_ids(ids_from, ids_selected):
	ids_from_inds = ids_from.argsort()
	ids_selected_inds = ids_selected.argsort()
	ids_from_sorted = np.sort(ids_from)
	ids_selected_sorted = np.sort(ids_selected)

	mask = np.zeros((len(ids_from)),dtype=bool)
	label_args = -np.ones((len(ids_from)),dtype=int) # Index of the label for that point
	index = 0
	for j, idv in enumerate(ids_selected_sorted):
		save_index = index
		while index < len(ids_from_sorted) and ids_from_sorted[index] < idv: #!= to <
			index += 1
		if index < len(ids_from_sorted) and ids_from_sorted[index] > idv:
			print('Iter',j,'of',len(ids_selected_sorted),'Id',idv,'not found')
			index = save_index #Since that idv value was not present
			continue
		while index < len(ids_from_sorted) and ids_from_sorted[index] == idv:
			mask[index] = True
			label_args[index] = ids_selected_inds[j]
			index += 1
		if index >= len(ids_from_sorted) and j < len(ids_selected_sorted)-1:
			print('Iter',j,'of',len(ids_selected_sorted),'Id',idv,'not found')
			index = save_index #Since that idv value was not present

	ids_from_extracted_inds = ids_from_inds[mask]
	ids_from_extracted_label_inds = label_args[mask]
	assert(not np.any(ids_from_extracted_label_inds == -1))
	return ids_from_extracted_inds, ids_from_extracted_label_inds

def align_or(idlist, labellist):
	all_ids = np.concatenate(idlist)
	all_labels = np.concatenate(labellist)
	id_inds = all_ids.argsort()
	all_labels = all_labels[id_inds]
	all_ids = all_ids[id_inds]
	num_unique = len(np.unique(all_ids))		

	merged_ids = np.empty((num_unique),dtype=object)
	merged_labels = -np.ones((num_unique),dtype=int)

	assert(len(all_labels)>0)
	assert(len(all_ids)==len(all_labels))
	assert(all_ids.shape == all_labels.shape)
	assert(len(all_ids.shape)==1)

	index = 0
	for j in range(num_unique):
		assert(index < len(all_ids))
		cur_id = all_ids[index]
		cur_label = all_labels[index]
		while index < len(all_ids) and all_ids[index] == cur_id:
			cur_label = (cur_label or all_labels[index])
			index += 1
		merged_ids[j] = cur_id
		merged_labels[j] = cur_label

	assert(not np.any(merged_labels == -1))	
	return merged_ids, merged_labels

# Assumes ref is already sorted so that days are near each other
def mean_encounter(ref, x, y, ref_cols):
	n = len(ref)
	print(ref_cols)
	newn = len(np.unique(ref[:,ref_cols['EncounterID']]))
	newx = scipy.sparse.lil_matrix((newn,x.shape[1]))
	newy = np.array((newn))
#	newref = np.array((newn,ref.shape[1]))
	newref_ids = np.zeros((newn))
	lower_bound = 0
	for j in range(newn):
		cur_id = ref[lower_bound,ref_cols['EncounterID']]
		upper_bound = lower_bound+1
		while upper_bound < newn and ref[upper_bound]==cur_id:
			upper_bound += 1
		newx[j,:] = x[lower_bound:upper_bound, :].mean(axis=0)
		newy[j] = y[lower_bound]
#		newref[j,:] = ref[lower_bound,:]
		newref_ids[j] = lower_bound
		lower_bound = upper_bound

	return newref_ids, scipy.sparse.csr_matrix(newx), newy

def extract_severe_subset_mean(trainref, xtrain, testref, xtest, refcols, ids_severe, labels_severe, refcols_severe, agg=['IHM', 'ODS', 'Readmit30Day'], refid = 'EncounterID'):
	print('Extracting severe subset of patients for training')
	print('Getting id and label lists')
	idlist = [ids_severe[:,refcols_severe[col]] for col in agg]
	labellist = [labels_severe[:,refcols_severe[col]] for col in agg]
	print('Merging severity labels')
	merged_ids, merged_labels = align_or(idlist, labellist)

	print(len(merged_ids), len(merged_labels))
#	assert(merged_ids[1] in trainref[:,refcols[refid]] or merged_ids[1] in testref[:,refcols[refid]])

	print('Stacking xtrain and xtest')
	x_all = scipy.sparse.vstack([xtrain,xtest])
	ref_all = np.vstack([trainref, testref])
	ids_from = ref_all[:,refcols[refid]]
	print('Extracting ids')
	ids_from_extracted_inds, ids_from_extracted_label_inds = extract_ids(ids_from, merged_ids)

	print('Constructing x_severe and y_severe')
	x_severe = x_all[ids_from_extracted_inds]
	ref_severe = ref_all[ids_from_extracted_inds,:]
	y_severe = merged_labels[ids_from_extracted_label_inds]

	print('Mean-aggregating')
	ref_severe_ids, x_severe, y_severe = mean_encounter(ref_severe, x_severe[:,:x_severe.shape[1]//7], y_severe, refcols)
	ref_severe = ref_severe[ref_severe_ids,:]

	print('Splitting into train and test')
	print(ref_severe[:5,refcols['AdmitDate']])
	train_severe_inds = ref_severe[:,refcols['AdmitDate']] < dt.datetime(2015,1,1)
	test_severe_inds = ~train_severe_inds
	xtrain_severe = x_severe[train_severe_inds, :]
	xtest_severe = x_severe[test_severe_inds,:]
	ytrain_severe = y_severe[train_severe_inds]
	ytest_severe = y_severe[test_severe_inds]
	trainref_severe = ref_severe[train_severe_inds,:]
	testref_severe = ref_severe[test_severe_inds,:]

	return xtrain_severe, xtest_severe, ytrain_severe, ytest_severe, trainref_severe, testref_severe
	
def extract_severe_subset(trainref, xtrain, testref, xtest, refcols, ids_severe, labels_severe, refcols_severe, agg=['IHM', 'ODS', 'Readmit30Day'], refid = 'EncounterID'):
	print('Extracting severe subset of patients for training')
	print('Getting id and label lists')
	idlist = [ids_severe[:,refcols_severe[col]] for col in agg]
	labellist = [labels_severe[:,refcols_severe[col]] for col in agg]
	print('Merging severity labels')
	merged_ids, merged_labels = align_or(idlist, labellist)

	print(len(merged_ids), len(merged_labels))
#	assert(merged_ids[1] in trainref[:,refcols[refid]] or merged_ids[1] in testref[:,refcols[refid]])

	print('Stacking xtrain and xtest')
	x_all = scipy.sparse.vstack([xtrain,xtest])
	ref_all = np.vstack([trainref, testref])
	ids_from = ref_all[:,refcols[refid]]
	print('Extracting ids')
	ids_from_extracted_inds, ids_from_extracted_label_inds = extract_ids(ids_from, merged_ids)

	print('Constructing x_severe and y_severe')
	x_severe = x_all[ids_from_extracted_inds]
	ref_severe = ref_all[ids_from_extracted_inds,:]
	y_severe = merged_labels[ids_from_extracted_label_inds]
	print('Splitting into train and test')
	train_severe_inds = ref_severe[:,refcols['AdmitDate']] < dt.datetime(2015,1,1)
	test_severe_inds = ~train_severe_inds
	xtrain_severe = x_severe[train_severe_inds, :]
	xtest_severe = x_severe[test_severe_inds,:]
	ytrain_severe = y_severe[train_severe_inds]
	ytest_severe = y_severe[test_severe_inds]
	trainref_severe = ref_severe[train_severe_inds,:]
	testref_severe = ref_severe[test_severe_inds,:]

	return xtrain_severe, xtest_severe, ytrain_severe, ytest_severe, trainref_severe, testref_severe
