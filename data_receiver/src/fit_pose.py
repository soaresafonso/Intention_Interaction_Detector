# import required libraries
from scipy import stats  
from scipy.stats import norm,rayleigh
from numpy import linspace
from pylab import plot,show,hist,figure,title
import numpy as np  
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import pickle
import math as m

# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

features = [[],[],[],[],[]]
target = [[],[],[],[],[]]
train = [[],[],[],[],[]]
# read the distances from the csv
train[0]=pd.read_csv("datasets/pose/posture1.csv", header=0, converters={"people/pose_keypoints_2d/6":float,"people/pose_keypoints_2d/8":float, "people/pose_keypoints_2d/17":float, "people/pose_keypoints_2d/15":float, "target":int})
train[1]=pd.read_csv("datasets/pose/posture2.csv", header=0, converters={"people/pose_keypoints_2d/6":float,"people/pose_keypoints_2d/8":float, "people/pose_keypoints_2d/17":float, "people/pose_keypoints_2d/15":float, "target":int})
train[2]=pd.read_csv("datasets/pose/pose3.csv", header=0, converters={"people/pose_keypoints_2d/6":float,"people/pose_keypoints_2d/8":float, "people/pose_keypoints_2d/17":float, "people/pose_keypoints_2d/15":float, "target":int})
train[3]=pd.read_csv("datasets/pose/posture4.csv", header=0, converters={"people/pose_keypoints_2d/6":float,"people/pose_keypoints_2d/8":float, "people/pose_keypoints_2d/17":float, "people/pose_keypoints_2d/15":float, "target":int})
train[4]=pd.read_csv("datasets/pose/posture5.csv", header=0, converters={"people/pose_keypoints_2d/6":float,"people/pose_keypoints_2d/8":float, "people/pose_keypoints_2d/17":float, "people/pose_keypoints_2d/15":float, "target":int})

d_shoulders = []
c_rigth = []
c_left = []
# obtain the the pose
for m in range(5):

    for i in range(len(train[m]['people/pose_keypoints_2d/15'])):

        if (train[m]['people/pose_keypoints_2d/15'][i] != 0 and train[m]['people/pose_keypoints_2d/6'][i] != 0 ):
            d_shoulders.append(abs(train[m]['people/pose_keypoints_2d/15'][i]-train[m]['people/pose_keypoints_2d/6'][i]))
            if train[m]['people/pose_keypoints_2d/8'][i] < 0.5:
                c_rigth.append(0)
            else:
                c_rigth.append(1)
            
            if train[m]['people/pose_keypoints_2d/17'][i] < 0.5:
                c_left.append(0)
            else:
                c_left.append(1)

            target[m].append(train[m]['target'][i])        

    # link the distances of shoulders and coffidence value for each shoulder
    #features=zip(d_shoulders, c_rigth, c_left)
    d_shoulders = np.reshape(d_shoulders,(-1,1))
    features[m]=list(d_shoulders)
    d_shoulders = []

# create the classifier for head pose using SVM
from sklearn.model_selection import cross_val_score
from sklearn import svm
i = 0.01
j = 0.01
a = 0
data1 = [[],[],[],[],[]]
data2 = [[],[],[],[],[]]
plt.figure(1)
plt.title('SVM Accuracy Scores')
plt.xlabel('Value of gamma for SVM')
plt.ylabel('Accuracy Score')
while j < 1000:
    while i < 1000:  
        clf = svm.SVC( gamma = i, C= j)
        scores = cross_val_score(clf, np.concatenate((features[0],features[1],features[2],features[3],features[4])), np.concatenate((target[0],target[1],target[2],target[3],target[4])), cv=5)
        accuracy = np.mean(scores)
        data1[a].append(i)
        data2[a].append(accuracy)
        i = i*10
    i = 0.01
    j = j*10
    a = a+1
line = plt.plot(data1[0], data2[0], '-bo',data1[1], data2[1], '-ro',data1[2], data2[2], '-yo',data1[3], data2[3], '-go',data1[4], data2[4], '-mo')
plt.legend((line[0], line[1], line[2], line[3], line[4]), ('C=0.01', 'C=0.1', 'C=1', 'C=10', 'C=100'))
plt.xscale('log')
plt.show()



# create the classifier for head pose using KNN
from sklearn.neighbors import KNeighborsClassifier
data1 = []
data2 = []
plt.figure(2)
plt.title('K Nearest Neighbors Accuracy Scores')
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy Score')
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, np.concatenate((features[0],features[1],features[2],features[3],features[4])), np.concatenate((target[0],target[1],target[2],target[3],target[4])), cv=5)
    accuracy = np.mean(scores)
    data1.append(i)
    data2.append(accuracy)
plt.plot(data1, data2, '-bo')
plt.show()

# choose best algorithm
clf = KNeighborsClassifier(n_neighbors = 9)
clf_final = clf.fit(np.concatenate((features[0],features[1],features[2],features[3],features[4])),np.concatenate((target[0],target[1],target[2],target[3],target[4])))



filename = 'pose_clf.sav'
pickle.dump(clf_final, open(filename, 'wb'))
