#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys,os,math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, auc, make_scorer
from scipy import stats
from numpy import interp
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from astropy.table import Table
import time


# Something strange seems to be happening in the execution of the inner and outer folds of the CV model, here I will investigate if I can understand what the difference is in this case.

# In[2]:


feat_df = Table.read('/home/xhall/Documents/PS1CrossMatch/PS1_star_galaxy_0.adamamiller_0.HST_COSMOS_Forcefeatures_xhall_0.fit').to_pandas()
ObjId = range(len(feat_df))
feat_df['ObjId'] = ObjId


# In[3]:


in_ts = np.where(feat_df["nDetections"] > 0)
feat_df = feat_df.iloc[in_ts]


# In[4]:


len(feat_df)


# In[5]:


feat_df.columns


# In[9]:


fnames = ['E1', 'E2', 'FPSFKronDist',
          'FPSFApDist', 'FPSFApRatio',  'FPSFKronRatio',
          'FPSFflxR5Ratio', 'FPSFflxR6Ratio', 'FPSFflxR7Ratio']
fil = 'ww'
features = [fil  + feat for feat in fnames]


# In[11]:


feat = feat_df.loc[:,features]

gt = (feat_df.MU_CLASS - 1).astype(int)
whiteKronMag = -2.5*np.log10(feat_df.wwFKronFlux/3631)

X = feat.values
y = np.squeeze(gt.values)


# In[12]:



from sklearn.metrics import make_scorer

def fom_score(y_true, y_pred, fpr_fom=0.005):
    """ZTF star-galaxy Figure of Merit (FoM) score.
    
    This metric calculates the true positive rate at a fixed
    false positive rate = 0.005. Assuming that stars are the 
    positive class, the objective is to maximize the FoM.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    
    Returns
    -------
    score : float
        The best performance is 1.
    """

    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    fom = interp(fpr_fom, fpr, tpr)
    return fom

fom_scorer = make_scorer(fom_score, needs_threshold=True)

grid = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                    param_grid={'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000],
                                'min_samples_leaf': [1, 2, 3],
                                'max_features': [2, 3, 4, 5]},
                    scoring=fom_scorer,
                    cv=10)


# In[10]:


rs = 23
N_outter_splits = 10
kf_cv = KFold(n_splits=N_outter_splits, shuffle=True, random_state=rs)

tuned_n_estimators = np.empty(N_outter_splits)
tuned_max_features = np.empty(N_outter_splits)
tuned_min_samples_leaf = np.empty(N_outter_splits)
fold_fom = np.empty(N_outter_splits)
fold_auc = np.empty(N_outter_splits)
fold_acu = np.empty(N_outter_splits)
interp_fpr = 10**(np.arange(-4, 0, 0.01))
interp_fpr = np.append(interp_fpr, 0.005)
interp_fpr = np.sort(interp_fpr)
interp_tpr = pd.DataFrame(index=range(len(interp_fpr)), columns=range(N_outter_splits))
CV_test_list = []
CV_proba_list = []

start = time.time()
print('Fold num: ')
for fold, (train, test) in zip(range(N_outter_splits), kf_cv.split(y)):
    print('{:d}/{:d}'.format(fold, N_outter_splits))
    grid.fit(X[train], y[train])
    if fold == 0:
        params_grid = grid.cv_results_['params']
        mean_test_score =  grid.cv_results_['mean_test_score']
    else: 
        mean_test_score = np.c_[mean_test_score, grid.cv_results_['mean_test_score']]
    tuned_param = grid.cv_results_['params'][np.argmin(grid.cv_results_['rank_test_score'])]
    tuned_n_estimators[fold] = tuned_param['n_estimators']
    tuned_max_features[fold] = tuned_param['max_features'] 
    tuned_min_samples_leaf[fold] = tuned_param['min_samples_leaf']
    
    best_model = RandomForestClassifier(n_estimators = tuned_param['n_estimators'], 
                                        min_samples_leaf = tuned_param['min_samples_leaf'], 
                                        max_features = tuned_param['max_features'], 
                                        n_jobs=-1)

    best_model.fit(X[train], y[train])
    predict = best_model.predict(X[test])
    proba = best_model.predict_proba(X[test])[:,1]
    CV_test_list.append(test)
    CV_proba_list.append( proba)
    fold_acu[fold] = accuracy_score(y[test], predict)
    fpr, tpr, _ = roc_curve(y[test], proba)
    fold_auc[fold] = auc(fpr, tpr)
    interp_tpr[fold] = interp(interp_fpr, fpr, tpr)
    fold_fom[fold] =  interp_tpr[interp_fpr==0.005][fold].values[0]
    elapsed_time = time.time() - start
    print('elapsed_time:{:.2f} [min]'.format(elapsed_time/60))


# In[11]:


mean_test_score_tab = pd.DataFrame(mean_test_score)
mean_test_score_tab


# In[12]:


mean_FoM = np.mean(mean_test_score_tab, axis=1)
std_FoM = np.std(mean_test_score_tab, axis=1)


# In[13]:


print('Mean FoM = {:.4f} +/- {:.4f}'.format(np.mean(fold_fom), np.std(fold_fom)))


# In[14]:


print('Optimal model params:')
print('\tN_tree = {:.1f}'.format(np.mean(tuned_n_estimators)))
print('\tm_try = {:.1f}'.format(np.mean(tuned_max_features)))
print('\tnodesize = {:.1f}'.format(np.mean(tuned_min_samples_leaf)))