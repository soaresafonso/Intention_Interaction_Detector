# import required libraries
import pandas as pd
import numpy as np
import pickle
import math as m
import matplotlib.pylab as plt
import hmmlearn.hmm as hmm

# receives the data from one gesture and obain the HMM for that gesture
def HMM_train(train):

    X=[]
    lengths = []

    # create train arrays for HMM
    for i in train:
        aux = []
        for j in range(len(train[i])):
            if not m.isnan(train[i][j]):
                aux.append(train[i][j])

        if len(aux) != 0:
            X.extend(aux)
            lengths.append(len(aux))

    lengths = np.array(lengths)
    X = np.array(X)
    X = X.astype(int)
    X = np.atleast_2d(X).T

    # HMM 
    remodel = hmm.MultinomialHMM(n_components=11, n_iter=10000000)

    remodel.fit(X, lengths)

    return remodel


# disable runtime warnings
np.seterr(divide='ignore')

# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

train = [[],[],[],[],[], []]
HMM_model = [[],[],[],[],[], []]


# read the data
train[0]=pd.read_csv("datasetes/handshake_rigth_gesture/handshake_r_sequences.csv", header=0)
train[1]=pd.read_csv("datasetes/handshake_left_gesture/handshake_l_sequences.csv", header=0)
train[2]=pd.read_csv("datasetes/hi_rigth_gesture/hi_r_sequences.csv", header=0)
train[3]=pd.read_csv("datasetes/handshake_left_gesture/hi_l_sequences.csv", header=0)
train[4]=pd.read_csv("datasetes/bow_gesture/bow_sequences.csv", header=0)
train[5]=pd.read_csv("datasetes/pray_gesture/pray_sequences.csv", header=0)

for i in range(len(HMM_model)):
    HMM_model[i] = HMM_train(train[i])


#HMM_model[0] = HMM_train(train[0])
#X = np.array([1])
#X = np.atleast_2d(X).T
#Z2 = HMM_model[0].decode(X, algorithm="viterbi")
#print(Z2)

#print(HMM_model[0].monitor_)
#print(remodel.score(X, lengths))

filename = 'HMM_handshake_r.sav'
pickle.dump(HMM_model[0], open(filename, 'wb'))

filename = 'HMM_handshake_l.sav'
pickle.dump(HMM_model[1], open(filename, 'wb'))

filename = 'HMM_hi_r.sav'
pickle.dump(HMM_model[2], open(filename, 'wb'))

filename = 'HMM_hi_l.sav'
pickle.dump(HMM_model[3], open(filename, 'wb'))

filename = 'HMM_bow.sav'
pickle.dump(HMM_model[4], open(filename, 'wb'))

filename = 'HMM_pray.sav'
pickle.dump(HMM_model[5], open(filename, 'wb'))
