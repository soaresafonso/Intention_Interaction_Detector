# import required libraries
import pandas as pd
import numpy as np
import pickle
import math as m
import matplotlib.pylab as plt
import hmmlearn.hmm as hmm

# disable runtime warnings
np.seterr(divide='ignore')

# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# read the data
train=pd.read_csv("sequences.csv", header=0)

X=[]
lengths = []

# create train arrays for HMM
for i in train:
    aux = []
    for j in range(len(train[i])):
        if not m.isnan(train[i][j]):
            aux.append([train[i][j]])

    if len(aux) != 0:
        X.extend(aux)
        lengths.append(len(aux))

#lengths = np.array(lengths)
#X = np.array(X)
#X = X.astype(int)
#X = np.atleast_2d(X).T

# HMM 
remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

remodel.fit(X, lengths)

X = [[1],[0]]
#X = np.atleast_2d(X).T
Z2 = remodel.predict(X)
print(Z2)

print(remodel.monitor_)
#print(remodel.score(X, lengths))

with open("HMM.pkl", "wb") as file: 
    pickle.dump(remodel, file)