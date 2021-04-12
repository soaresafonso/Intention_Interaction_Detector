# import required libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt

# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# read the data
train=pd.read_csv("datasets/head_pose/fit_headpose15.csv", header=0, converters={"x_angle":float, "y_angle":float, "z_angle":float, "classification":int})

# obtain the the angles for the head pose
x_angle = train['x_angle']
y_angle = train['y_angle']
z_angle = train['z_angle']
target= train['classification']

# link the x_angle , y_angle and z_angle features
feat=zip(x_angle, y_angle, z_angle)
features=list(feat)

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
        scores = cross_val_score(clf, features, target, cv=5)
        accuracy = np.mean(scores)
        data1[a].append(i)
        data2[a].append(accuracy)
        i = i*10
    i = 0.01
    j = j*10
    a = a+1
line = plt.plot(data1[0], data2[0], '-bo',data1[1], data2[1], '-ro',data1[2], data2[2], '-yo',data1[3], data2[3], '-go',data1[4], data2[4], '-mo')
plt.legend((line[0], line[1], line[2], line[3], line[4]), ('C=0.01', 'C=0.1', 'C=1', 'C=10', 'C=100'), loc='lower right')
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
    scores = cross_val_score(knn, features, target, cv=5)
    accuracy = np.mean(scores)
    data1.append(i)
    data2.append(accuracy)
plt.plot(data1, data2, '-bo')
plt.show()


# choose best algorithm
clf = svm.SVC( gamma = 10, C = 1, probability=True)
clf_final = clf.fit(features,target)


filename = 'headpose_clf.sav'
pickle.dump(clf_final, open(filename, 'wb'))
