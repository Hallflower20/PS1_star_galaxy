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
import astropy
import glob
import pickle

print("Started")

fnames = ['objid', 'wwE1',
 'wwE2',
 'wwFPSFKronDist',
 'wwFPSFApDist',
 'wwFPSFApRatio',
 'wwFPSFKronRatio',
 'wwFPSFflxR5Ratio',
 'wwFPSFflxR6Ratio',
 'wwFPSFflxR7Ratio']
features = ['wwE1',
 'wwE2',
 'wwFPSFKronDist',
 'wwFPSFApDist',
 'wwFPSFApRatio',
 'wwFPSFKronRatio',
 'wwFPSFflxR5Ratio',
 'wwFPSFflxR6Ratio',
 'wwFPSFflxR7Ratio']

data_dir = "/home/xhall/Documents/PS1_MLData/"
data_files = np.sort(glob.glob(data_dir + "/*.fit"))

classifier = pickle.load(open("model.model", 'rb'))
mldata_nona = pd.read_hdf(data_dir + "PS1_MLData.hdf")

print("Read In")

objids = np.asarray(mldata_nona["objid"])
X_hst = np.asarray(mldata_nona[features])

print("Predictions Started")

predictions = classifier.predict_proba(X_hst)[:,1]

print("Predictions Done")

df_out = pd.DataFrame(data=objids, columns=["objid"])
df_out["rf_score"] = predictions
df_out.to_hdf(data_dir + "class_table.hdf", "class_table")

print("Predictions Outputed")