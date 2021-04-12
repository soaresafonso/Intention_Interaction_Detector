import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


from sklearn.metrics import confusion_matrix
import seaborn as sns

# load segment classificator
filename = 'gestures_model.sav'
segment_clf = pickle.load(open(filename, 'rb'))

# load HMMs 
filename = 'HMM_handshake_r.sav'
HMM_model_handr = pickle.load(open(filename, 'rb'))
filename = 'HMM_handshake_l.sav'
HMM_model_handl = pickle.load(open(filename, 'rb'))
filename = 'HMM_hi_r.sav'
HMM_model_hir = pickle.load(open(filename, 'rb'))
filename = 'HMM_hi_l.sav'
HMM_model_hil = pickle.load(open(filename, 'rb'))
filename = 'HMM_bow.sav'
HMM_model_bow = pickle.load(open(filename, 'rb'))
filename = 'HMM_pray.sav'
HMM_model_pray = pickle.load(open(filename, 'rb'))
filename = 'HMM_nogest.sav'
HMM_model_nogest = pickle.load(open(filename, 'rb'))

y_pred = []
feature = [[],[],[],[],[],[]]
# read the data
train1=pd.read_csv("datasets/gestures_detection/gest_conf.csv", header=0, converters={"classes":int})
train2=pd.read_csv("datasets/gestures_detection/seq_conf.csv", header=0, converters={"first":int,"second":int,"third":int,"fourth":int,"fifth":int,"sixth":int}) 

# obtain the features and classification
feature[0] = train2['first']
feature[1] = train2['second']
feature[2] = train2['third']
feature[3] = train2['fourth']
feature[4] = train2['fifth']
feature[5] = train2['sixth']
target = train1['classes']

class_names = ['handshake_r', 'handshake_l', 'wave_r', 'wave_l', 'pray', 'bow', 'no_gesture']

for i in range(len(feature[0])):


    if feature[2][i] == 0 and feature[3][i] == 0 and feature[4][i] == 0 and  feature[5][i] == 0:
        sequence = [feature[0][i],feature[1][i]]
    
    elif feature[3][i] == 0 and feature[4][i] == 0 and  feature[5][i] == 0:
        sequence = [feature[0][i],feature[1][i], feature[2][i]]
    
    elif feature[4][i] == 0 and  feature[5][i] == 0:
        sequence = [feature[0][i],feature[1][i], feature[2][i], feature[3][i]]
    
    elif feature[5][i] == 0:
        sequence = [feature[0][i],feature[1][i], feature[2][i], feature[3][i], feature[4][i]]

    else:
        sequence = [feature[0][i],feature[1][i],feature[2][i],feature[3][i],feature[4][i], feature[5][i]]

    if len(sequence) < 3 and  feature[0][i] == 0 and feature[1][i] == 0:
        y_pred.extend([6])
    else :
    
        if i > 80:
            sequence.extend([0])

        X = np.array([sequence])
        X = np.atleast_2d(X).T

        Z1 = HMM_model_handr.decode(X, algorithm="viterbi")

        Z2 = HMM_model_handl.decode(X, algorithm="viterbi")

        Z3 = HMM_model_hir.decode(X, algorithm="viterbi")

        Z4 = HMM_model_hil.decode(X, algorithm="viterbi")

        Z6 = HMM_model_bow.decode(X, algorithm="viterbi")

        Z5 = HMM_model_pray.decode(X, algorithm="viterbi")

        prob = [-Z1[0], -Z2[0], -Z3[0], -Z4[0], -Z5[0], -Z6[0]]
        ind = prob.index(min(prob))

        if ind == 0:
            y_pred.extend([0])
        if ind == 1:
            y_pred.extend([1])
        if ind == 2:
            y_pred.extend([2])

        if ind == 3:
            y_pred.extend([3])

        if ind == 4:
            y_pred.extend([4])

        if ind == 5:
            y_pred.extend([5])


target = list(target)
print(y_pred)
print(target)
cm = confusion_matrix(target, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=2)#for label size
sns.heatmap(cmn, annot=True, annot_kws={"size": 20}, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Gesture')
plt.xlabel('Predicted Gesture')
plt.show()
