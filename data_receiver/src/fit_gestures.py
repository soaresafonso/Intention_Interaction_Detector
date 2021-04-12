# import required libraries
from scipy import stats  
from scipy.stats import norm,rayleigh
from numpy import linspace
from pylab import plot,show,hist,figure,title
import numpy as np  
import matplotlib.pylab as plt
import pandas as pd
import pickle
import math as m
from math import pow
from scipy.signal import argrelextrema
from sklearn.neighbors import KNeighborsClassifier
import warnings


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    for i in range(len(y)-1):
        if y[i] != 0:
            if (y[i+1] == y[i]):
                y[i+1] = y[i+1]+1

    return y



warnings.filterwarnings("ignore")


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



# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

# read the distances from the csv
train=pd.read_csv("hi_r3.csv", header=0, usecols=["people/pose_keypoints_2d/0","people/pose_keypoints_2d/1", "people/pose_keypoints_2d/3", "people/pose_keypoints_2d/4", "people/pose_keypoints_2d/6", "people/pose_keypoints_2d/7", "people/pose_keypoints_2d/9", "people/pose_keypoints_2d/10", "people/pose_keypoints_2d/12", "people/pose_keypoints_2d/13", "people/pose_keypoints_2d/15", "people/pose_keypoints_2d/16", "people/pose_keypoints_2d/18", "people/pose_keypoints_2d/19", "people/pose_keypoints_2d/21", "people/pose_keypoints_2d/22","people/pose_keypoints_2d/24", "people/pose_keypoints_2d/25", "time"], converters={"people/pose_keypoints_2d/0":float,"people/pose_keypoints_2d/1":float, "people/pose_keypoints_2d/3":float, "people/pose_keypoints_2d/4":float, "people/pose_keypoints_2d/6":float, "people/pose_keypoints_2d/7":float, "people/pose_keypoints_2d/9":float, "people/pose_keypoints_2d/10":float, "people/pose_keypoints_2d/12":float, "people/pose_keypoints_2d/13":float, "people/pose_keypoints_2d/15":float, "people/pose_keypoints_2d/16":float, "people/pose_keypoints_2d/18":float, "people/pose_keypoints_2d/19":float, "people/pose_keypoints_2d/21":float, "people/pose_keypoints_2d/22":float, "people/pose_keypoints_2d/24":float, "people/pose_keypoints_2d/25":float, "time":float})

size = len(train)
ac= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [], [], []]
h = 0


# obtain the acceleration for x and y
for i in train:
    if (h <= 17):
        for j in range(len(train["people/pose_keypoints_2d/0"])-1):

            if i == "people/pose_keypoints_2d/0" or i == "people/pose_keypoints_2d/1":

                if train[i][j] == 0 or train[i][j+1] == 0:
                    ac[h].append(0)

                elif m.fabs(train[i][j+1]-train[i][j]) > 20:
                    ac[h].append((train[i][j+1]-train[i][j])/(train["time"][j+1]-train["time"][j]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(train["time"][j+1]-train["time"][j])
                else:
                    ac[h].append(0)
                    
            else:

                if train[i][j] == 0 or train[i][j+1] == 0:
                    ac[h].append(0)

                elif m.fabs(train[i][j+1]-train[i][j]) > 30:
                    ac[h].append((train[i][j+1]-train[i][j])/(train["time"][j+1]-train["time"][j]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(train["time"][j+1]-train["time"][j])
                else:
                    ac[h].append(0)
    h = h+1        


ac_ = [[],[],[],[],[],[],[],[], []]
count = 0
i = 0
windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

# obtain the acceleration for each 2d coordinates 
while i <= 17:

        for j in range(len(ac[i])):
            ac_[count].append(m.sqrt(pow(ac[i][j],2)+pow(ac[i+1][j],2)))

        ac_[count] = np.array(ac_[count], dtype=float)
        #print(ac_[count])    
        ac_[count] = smooth(ac_[count], 10, 'bartlett')
        #print(ac_[count])    
        np.gradient(ac_[count])

        count = count+1
        i = i+2




seg = [[],[],[],[],[],[],[],[], []]
aux = [[],[],[],[],[],[],[],[], []]

#segment the trajectory
for i in range(len(ac_)):
    
    # for local maxima
    Max = argrelextrema(ac_[i], np.greater)
    # for local minima
    Min = argrelextrema(ac_[i], np.less)
    Max = list(Max[0])
    Min = list(Min[0])
    #print(Max)
    #print(Min)

    # find the spots to segment the movement
    if len(Max) > 0:
        for j in range(len(ac_[i])-1):
            if (ac_[i][j] == 0 and ac_[i][j+1] != 0):
                seg[i].append(j+1)
                aux[i].append(j+1)
            elif (ac_[i][j] != 0 and ac_[i][j+1] == 0):
                seg[i].append(j)
                aux[i].append(j)

        seg[i].extend(Max)

        if len(Min) > 0:
            seg[i].extend(Min)

        seg[i].sort(reverse=False)

#print(seg)
feature = [[],[],[],[],[]]

# obtain the direction of segmented movement
for i in range(len(seg)):
    if len(seg[i]):
        j=0
        joint = i*3
        while j < len(seg[i])-2:
            
            z1 = "people/pose_keypoints_2d/%s" % joint
            first_point_x = train[z1][seg[i][j]]

            if z1 == "people/pose_keypoints_2d/12":
                first_point_x_1 = first_point_x-train["people/pose_keypoints_2d/6"][seg[i][j]]
                first_point_x_2 = first_point_x-train["people/pose_keypoints_2d/9"][seg[i][j]] 

            if z1 == "people/pose_keypoints_2d/21":
                first_point_x_1 = first_point_x-train["people/pose_keypoints_2d/15"][seg[i][j]]
                first_point_x_2 = first_point_x-train["people/pose_keypoints_2d/18"][seg[i][j]]


            if z1 == "people/pose_keypoints_2d/0":
                first_point_x_1 = first_point_x-train["people/pose_keypoints_2d/24"][seg[i][j]]

            joint_ = joint + 1
            z2 = "people/pose_keypoints_2d/%s" % joint_

            first_point_y = train[z2][seg[i][j]]
            
            if z2 == "people/pose_keypoints_2d/13":
                first_point_y_1 = first_point_y-train["people/pose_keypoints_2d/7"][seg[i][j]]
                first_point_y_2 = first_point_y-train["people/pose_keypoints_2d/10"][seg[i][j]] 

            if z2 == "people/pose_keypoints_2d/22":
                first_point_y_1 = first_point_y-train["people/pose_keypoints_2d/16"][seg[i][j]]
                first_point_y_2 = first_point_y-train["people/pose_keypoints_2d/19"][seg[i][j]]
            
            if z2 == "people/pose_keypoints_2d/1":
                first_point_y_1 = first_point_y-train["people/pose_keypoints_2d/25"][seg[i][j]]

            j = j+2
            second_point_x = train[z1][seg[i][j]]
            second_point_y = train[z2][seg[i][j]]

            if z1 == "people/pose_keypoints_2d/12":
                second_point_x_1 = second_point_x-train["people/pose_keypoints_2d/6"][seg[i][j]]
                second_point_x_2 = second_point_x-train["people/pose_keypoints_2d/9"][seg[i][j]] 
            
            if z2 == "people/pose_keypoints_2d/13":
                second_point_y_1 = second_point_y-train["people/pose_keypoints_2d/7"][seg[i][j]]
                second_point_y_2 = second_point_y-train["people/pose_keypoints_2d/10"][seg[i][j]] 
            
            if z1 == "people/pose_keypoints_2d/21":
                second_point_x_1 = second_point_x-train["people/pose_keypoints_2d/15"][seg[i][j]]
                second_point_x_2 = second_point_x-train["people/pose_keypoints_2d/18"][seg[i][j]] 
            
            if z2 == "people/pose_keypoints_2d/22":
                second_point_y_1 = second_point_y-train["people/pose_keypoints_2d/16"][seg[i][j]]
                second_point_y_2 = second_point_y-train["people/pose_keypoints_2d/19"][seg[i][j]] 

            if z1 == "people/pose_keypoints_2d/0":
                second_point_x_1 = second_point_x-train["people/pose_keypoints_2d/24"][seg[i][j]]
            
            if z2 == "people/pose_keypoints_2d/1":
                second_point_y_1 = second_point_y-train["people/pose_keypoints_2d/25"][seg[i][j]]

            if seg[i][j] in aux[i]:
                end_point = 1
            else:
                end_point = 0
            
            # conditions for bow gesture
            if z1 == "people/pose_keypoints_2d/0":

                # normalize the points
                dist1 = m.sqrt(m.pow(first_point_x_1,2)+m.pow(first_point_y_1,2))
                dist2 = m.sqrt(m.pow(second_point_x_1,2)+m.pow(second_point_y_1,2))
                
                first_point_x_1 = first_point_x_1/dist1
                first_point_y_1 = first_point_y_1/dist1
                
                second_point_x_1 = second_point_x_1/dist2
                second_point_y_1 = second_point_y_1/dist2
                
                # find angle for neck-trunk
                if second_point_y_1 >= first_point_y_1:
        
                    if (second_point_y_1 <= 0):
                        angle_lambda = -(m.acos(second_point_y_1/first_point_y_1))

                    else:
                        angle_lambda = -(m.pi - m.acos((-second_point_y_1/first_point_y_1)))

                else:
                    
                    if (second_point_y_1 <= 0):
                        angle_lambda = (m.acos(first_point_y_1/second_point_y_1))

                    else:
                        angle_lambda = (m.pi - m.acos((first_point_y_1*(-1))/second_point_y_1))

                angle_lambda = round(m.degrees(angle_lambda),2)
                feature[4].extend([angle_lambda])


            # conditions for handshake, hi and praying gestures
            if z1 == "people/pose_keypoints_2d/12" or z1 == "people/pose_keypoints_2d/21":
                
                # normalize the points
                dist1 = m.sqrt(m.pow(first_point_x_1,2)+m.pow(first_point_y_1,2))
                dist2 = m.sqrt(m.pow(first_point_x_2,2)+m.pow(first_point_y_2,2))
                dist3 = m.sqrt(m.pow(second_point_x_1,2)+m.pow(second_point_y_1,2))
                dist4 = m.sqrt(m.pow(second_point_x_2,2)+m.pow(second_point_y_2,2))
                
                first_point_x_1 = first_point_x_1/dist1
                first_point_y_1 = first_point_y_1/dist1
                first_point_x_2 = first_point_x_2/dist2
                first_point_x_2 = first_point_y_2/dist2

                second_point_x_1 = second_point_x_1/dist3
                second_point_y_1 = second_point_y_1/dist3
                second_point_x_2 = second_point_x_2/dist4
                second_point_y_2 = second_point_y_2/dist4
                
                # find angle for hand-soulder
                if (first_point_x_1) != 0:

                    # find the slope
                    slope = (first_point_y_1)/(first_point_x_1)

                    if first_point_x_1 < 0:
                        slope = slope*(-1)
                        #print(slope)
                    # find the angle
                    angle_1 = m.atan(slope)

                else:

                    if first_point_y_1 <= 0:
                        angle_1 = -(m.pi)/2

                    else:
                        angle_1 = (m.pi)/2

                if (second_point_x_1) != 0:

                    # find the slope
                    slope = (second_point_y_1)/(second_point_x_1)

                    if second_point_x_1 < 0:
                        slope = slope*(-1)
                    
                    # find the angle
                    angle_2 = m.atan(slope)

                else:

                    if second_point_y_1 <= 0:
                        angle_2 = (m.pi)/2

                    else:
                        angle_2 = -(m.pi)/2

                angle_shoulder = angle_2-angle_1
                angle = round(m.degrees(angle_shoulder),2)
                
                if z1 == "people/pose_keypoints_2d/12":
                    feature[0].extend([angle])

                if z1 == "people/pose_keypoints_2d/21":
                    feature[1].extend([angle])

                # find angle for hand-elbow
                if first_point_y_2 < 0:
                    first_point_y_2 = first_point_y_2*(-1)
                    
                # find the angle
                angle_1 = m.atan2(first_point_y_2,first_point_x_2)
    
                if second_point_y_2 < 0:
                    second_point_y_2 = second_point_y_2*(-1)
                   
                # find the angle
                angle_2 = m.atan2(second_point_y_2,second_point_x_2)

                angle_elbow = angle_2-angle_1
                angle = round(m.degrees(angle_elbow),2)
                
                if z1 == "people/pose_keypoints_2d/12":
                    feature[2].extend([angle])

                if z1 == "people/pose_keypoints_2d/21":
                    feature[3].extend([angle])
                
            
            j= j+end_point

n = 0
m = 0

for i in range(len(feature)):
    n = len(feature[i])
    if n > m:
        m = n

for i in range(len(feature)):
    zeros = m - len(feature[i])
    if zeros != 0:
        v = np.zeros(zeros)
        feature[i].extend(v)

print(feature)

# obtain the x and y distance of the eye gaze
plt.figure(1)
title('Distribution of the looking points in 2D')
plt.xlabel('Distance along x')
plt.ylabel('Distance along y')
x = train['people/pose_keypoints_2d/12']
y = train['people/pose_keypoints_2d/13']
plt.scatter(x, y, color = 'red')
plt.show()

sequence = []
for i in range(len(feature[0])):

    values = [feature[0][i],feature[1][i],feature[2][i],feature[3][i],feature[4][i]]
    sequence.extend(list(segment_clf.predict([values])))

print(sequence)

X = np.array([sequence])
X = np.atleast_2d(X).T


Z1 = HMM_model_handr.decode(X, algorithm="viterbi")

Z2 = HMM_model_handl.decode(X, algorithm="viterbi")

Z3 = HMM_model_hir.decode(X, algorithm="viterbi")

Z4 = HMM_model_hil.decode(X, algorithm="viterbi")

Z5 = HMM_model_bow.decode(X, algorithm="viterbi")

Z6 = HMM_model_pray.decode(X, algorithm="viterbi")


prob = [-Z1[0], -Z2[0], -Z3[0], -Z4[0], -Z5[0], -Z6[0]]
print(prob)
ind = prob.index(min(prob))

if ind == 0:
    print("HANDSHAKE RIGHT HAND!!")

if ind == 1:
    print("HANDSHAKE LEFT HAND!!")

if ind == 2:
    print("HI RIGHT HAND!!")

if ind == 3:
    print("HI LEFT HAND!!")

if ind == 4:
    print("DO A BOW!!")

if ind == 5:
    print("PRAY!!")
