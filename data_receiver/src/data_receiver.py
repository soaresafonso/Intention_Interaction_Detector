#!/usr/bin/env python3
#!/usr/bin/env python
import rospy
#import cv2
import numpy as np
from openface2_ros.msg import Faces
from openpose_ros_msgs.msg import OpenPoseHumanList
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
from getkey import getkey, keys
import pandas as pd
import pickle
import math as m
from numpy import linspace
from math import pow
from scipy.signal import argrelextrema
from pandas import DataFrame
import warnings
from time import sleep

# Interface for the intention of interaction and the gestures detection


# Global variables
#image = cv2.imread('nointeraction.jpeg')
#image2 = cv2.imread('nointeraction.jpeg')
OpenFace_sub = False
OpenPose_sub = False
Interaction = 0
x = 0
y = 0
d = 0
d_pose = 0
time = 0
h_angle = []
pose = []
gestures = 0
end_gesture = 0
count2 = 0

# create new dictionaries
data1={}
data1['Distances'] = []
data1['x'] = []
data1['y'] = []
data2={}
data2['x_angle'] = []
data2['y_angle'] = []
data2['z_angle'] = []
data2['classification'] = []
data3={}
data3['Intent_prob'] = []

# load headpose classificator
filename = 'headpose_clf.sav'
headpose_clf = pickle.load(open(filename, 'rb'))

# load pose classificatior
filename = 'pose_clf.sav'
pose_clf = pickle.load(open(filename, 'rb'))

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

warnings.filterwarnings("ignore")

# ros callback to read the data from openface
def openface_callback(msg):

    global OpenPose_sub
    global OpenFace_sub
    global x
    global y
    global d
    global h_angle

    if (len(msg.faces) >= 1):

        # read the data
        left_gaze = msg.faces[0].left_gaze.position
        right_gaze = msg.faces[0].right_gaze.position
        h_p = msg.faces[0].head_pose.position
        h_p_q = msg.faces[0].head_pose.orientation

        h_p.x = h_p.x/1000
        h_p.y = h_p.y/1000
        h_p.z = h_p.z/1000

        # obtain the rotation matrix of the camera
        # obtain the Y and X matrices
        alfa = m.atan(h_p.x/h_p.z)
        R_Y = np.array([[m.cos(alfa), 0, m.sin(alfa)], [0,1,0], [-m.sin(alfa), 0, m.cos(alfa)]])
        z_n = h_p.z/m.cos(alfa)
        beta = m.atan(h_p.y/z_n)
        R_X = np.array([[1,0,0], [0, m.cos(beta), -m.sin(beta)], [0, m.sin(beta), m.cos(beta)]])

        # obtain the matrix given from the ros quaternion
        r = R.from_quat([h_p_q.x, h_p_q.y, h_p_q.z, h_p_q.w])
        R_o = r.as_dcm()

        # obtain the final rotation matrix for head
        R_f = np.dot(R_Y, np.dot( R_X, R_o ))
        # obtain the head_pose vector angle
        h_angle = rotationMatrixToEulerAngles(R_f)
        # rotate the head_pose vector
        h_p_f = rot_euler( [h_p.x, h_p.y, h_p.z] , h_angle)

        # normalize vector
        h_p_f = h_p_f/np.linalg.norm(h_p_f)

        #print(l_angle)
        # obtain the final rotation matrix for eye
        R_f_e = np.dot(R_Y,R_X)
        # obtain the head_pose vector angle
        eye_angle = rotationMatrixToEulerAngles(R_f_e)
        # rotate the head_pose vector
        h_p_e = rot_euler( [h_p.x, h_p.y, h_p.z] , eye_angle)

        # normalize vector
        h_p_e = h_p_e/np.linalg.norm(h_p_e)

        # obtain the new euler angles for the new vector and invert this vector
        h_a = []
        h_a.append(m.acos(-h_p_f[0]))
        h_a.append(m.acos(-h_p_f[1]))
        h_a.append(m.acos(-h_p_f[2]))
        # obtain the new euler angles for the new vector and invert this vector
        e_angle = []
        e_angle.append(m.acos(-h_p_e[0]))
        e_angle.append(m.acos(-h_p_e[1]))
        
        '''
        # angles for left eye
        n = m.sqrt(m.pow(left_gaze.x,2)+m.pow(left_gaze.y,2))
        if n == 0:
            x_angle_l = 0
            y_angle_l = 0
        else:   
            left_gaze_x = left_gaze.x/n
            left_gaze_y = left_gaze.y/n
            x_angle_l = m.acos(left_gaze_x)
            y_angle_l = m.acos(left_gaze_y)

        # angles for right eye
        n = m.sqrt(m.pow(right_gaze.x,2)+m.pow(right_gaze.y,2))
        if n == 0:
            x_angle_r = 0
            y_angle_r = 0
        else:
            right_gaze_x = right_gaze.x/n
            right_gaze_y = right_gaze.y/n
            x_angle_r = m.acos(right_gaze_x)
            y_angle_r = m.acos(right_gaze_y)
        
        # mean angle of both eyes
        x_angle = (x_angle_l+x_angle_r)/2
        y_angle = (y_angle_l+y_angle_r)/2

        '''
        # final eye gaze angle
        x_angle = e_angle[0]-h_a[0]
        y_angle = e_angle[1]-h_a[1]

        # obtain the distance from the center of the camera to the end of the eye gaze vector
        x = m.tan(x_angle)*h_p.z
        y = m.tan(y_angle)*h_p.z
        d = float(np.linalg.norm([x,y]))

        #DataSaving(x,y,d,h_angle)

        OpenFace_sub = True

        if ( OpenFace_sub and OpenPose_sub):
            Intent_int()
            OpenFace_sub = False
            OpenPose_sub = False
            
# ros callback to read the data from openpose
def openpose_callback(msg):

    global OpenPose_sub
    global OpenFace_sub
    global d_pose
    global pose
    global gestures
    global Interaction
    global time
    global end_gesture
    global count2
    global image2
    global image

    if len(msg.human_list) >= 1:
        # read the data
        body = msg.human_list[0].body_key_points_with_prob
        # posture for interaction
    
        if (body[5].x != 0 and body[2].x != 0):

            # find the distance from right shoulder to left
            rightshoulder = body[2].x
            leftshoulder = body[5].x

            d_pose = abs((leftshoulder-rightshoulder))

            OpenPose_sub = True

            if ( OpenFace_sub and OpenPose_sub):
                Intent_int()
                OpenFace_sub = False
                OpenPose_sub = False
        # gestures
        if Interaction == 1:
            if len(pose) == 0:
                body[9] = time
                pose.append(body[0:10])
            else:
                time = time+0.1
                body[9] = time
                aux = body[0:10]
                pose.append(aux)
                if (1):
                    h = 0
                    ac= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [], [], []]
                    # obtain the acceleration for x and y
                    for i in range(9):
                        for j in range(len(pose)-1):

                            if i == 0:
                                
                                h = i*2
                                if pose[j][i].x == 0 or pose[j+1][i].x == 0:
                                    ac[h].append(0)

                                elif m.fabs(pose[j][i].x-pose[j+1][i].x) > 20:
                                    ac[h].append((pose[j+1][i].x-pose[j][i].x)/(pose[j+1][9]-pose[j][9]))
                                    l = len(ac[h])
                                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                                else:
                                    ac[h].append(0)

                                h = h+1
                                
                                if pose[j][i].y == 0 or pose[j+1][i].y == 0:
                                    ac[h].append(0)

                                elif m.fabs(pose[j][i].y-pose[j+1][i].y) > 20:
                                    ac[h].append((pose[j+1][i].y-pose[j][i].y)/(pose[j+1][9]-pose[j][9]))
                                    l = len(ac[h])
                                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                                else:
                                    ac[h].append(0)
                            else:
                                
                                h = i*2

                                if pose[j][i].x == 0 or pose[j+1][i].x == 0:
                                    ac[h].append(0)

                                elif m.fabs(pose[j][i].x-pose[j+1][i].x) > 30:
                                    ac[h].append((pose[j+1][i].x-pose[j][i].x)/(pose[j+1][9]-pose[j][9]))
                                    l = len(ac[h])
                                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                                else:
                                    ac[h].append(0)

                                h = h+1
                                
                                if pose[j][i].y == 0 or pose[j+1][i].y == 0:
                                    ac[h].append(0)

                                elif m.fabs(pose[j][i].y-pose[j+1][i].y) > 30:
                                    ac[h].append((pose[j+1][i].y-pose[j][i].y)/(pose[j+1][9]-pose[j][9]))
                                    l = len(ac[h])
                                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                                else:
                                    ac[h].append(0)

                    for i in ac[:]:
                        if i != 0:
                            gestures = 1
                   
                    if gestures == 0:
                        pose = []
                        time = 0 

                if gestures == 1:

                    count2 = count2 + 1
                    for i in range(len(ac)):
                        if i == 0 or i == 4 or i == 7:
                            if ac[i][-1] != 0:
                                count2 = 0
                    #print(count)
                    if count2 >= 20:
                        end_gesture=1
                        count2=0

                if end_gesture == 1:
                    feature = gestures_clf(pose)
                    gestures = 0
                    count2 = 0
                    pose = []
                    end_gesture = 0

                    if len(feature[0]) >= 1:
                        sequence = []
                        sequence_aux = [0]
                        for i in range(len(feature[0])):

                            values = [feature[0][i],feature[1][i],feature[2][i],feature[3][i],feature[4][i]]
                            if feature[0][i] == 0 and feature[1][i] == 0 and feature[2][i] == 0 and feature[3][i] == 0 and feature[4][i] == 0:
                                sequence.extend([0])
                            else:
                                sequence.extend(list(segment_clf.predict([values])))

                        #print(sequence)
                        if sequence != sequence_aux:
                            X = np.array([sequence])
                            X = np.atleast_2d(X).T


                            Z1 = HMM_model_handr.decode(X, algorithm="viterbi")

                            Z2 = HMM_model_handl.decode(X, algorithm="viterbi")

                            Z3 = HMM_model_hir.decode(X, algorithm="viterbi")

                            Z4 = HMM_model_hil.decode(X, algorithm="viterbi")

                            Z5 = HMM_model_bow.decode(X, algorithm="viterbi")

                            Z6 = HMM_model_pray.decode(X, algorithm="viterbi")

                            prob = [-Z1[0], -Z2[0], -Z3[0], -Z4[0], -Z5[0], -Z6[0]]
                            ind = prob.index(min(prob))

                            if ind == 0:
                                print("HANDSHAKE RIGHT HAND!!")
                                #image2 = cv2.imread('handshake.png')

                            if ind == 1:
                                print("HANDSHAKE LEFT HAND!!")
                                #image2 = cv2.imread('handshake.png')

                            if ind == 2:
                                print("HI RIGHT HAND!!")
                                #image2 = cv2.imread('wave.png')

                            if ind == 3:
                                print("HI LEFT HAND!!")
                                #image2 = cv2.imread('wave.png')

                            if ind == 4:
                                print("DO A BOW!!")
                                #image2 = cv2.imread('bow.jpg')

                            if ind == 5:
                                print("PRAY !!")
                                #image2 = cv2.imread('pray.jpg')

                            #numpy_horizontal_concat = np.concatenate((image, image2), axis=1)

                            #cv2.imshow('image', numpy_horizontal_concat)
                            #cv2.moveWindow('image', 2000,0) 


# Rotate vector v (or array of vectors) by the euler angles xyz
def rot_euler(v, xyz):
    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis/norm(axis)*theta)))
    return v



# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Segment gesture and classify it
def gestures_clf(pose):

    h = 0
    ac= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [], [], []]
    # obtain the acceleration for x and y
    for i in range(9):
        for j in range(len(pose)-1):

            if i == 0:

                h = i*2
                if pose[j][i].x == 0 or pose[j+1][i].x == 0:
                    ac[h].append(0)

                elif m.fabs(pose[j][i].x-pose[j+1][i].x) > 20:
                    ac[h].append((pose[j+1][i].x-pose[j][i].x)/(pose[j+1][9]-pose[j][9]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                else:
                    ac[h].append(0)

                h = h+1
                                
                if pose[j][i].y == 0 or pose[j+1][i].y == 0:
                    ac[h].append(0)

                elif m.fabs(pose[j][i].y-pose[j+1][i].y) > 20:
                    ac[h].append((pose[j+1][i].y-pose[j][i].y)/(pose[j+1][9]-pose[j][9]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                else:
                    ac[h].append(0)
                            
            else:

                h = i*2
                if pose[j][i].x == 0 or pose[j+1][i].x == 0:
                    ac[h].append(0)

                elif m.fabs(pose[j][i].x-pose[j+1][i].x) > 30:
                    ac[h].append((pose[j+1][i].x-pose[j][i].x)/(pose[j+1][9]-pose[j][9]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                else:
                    ac[h].append(0)

                h = h+1
                                
                if pose[j][i].y == 0 or pose[j+1][i].y == 0:
                    ac[h].append(0)

                elif m.fabs(pose[j][i].y-pose[j+1][i].y) > 30:
                    ac[h].append((pose[j+1][i].y-pose[j][i].y)/(pose[j+1][9]-pose[j][9]))
                    l = len(ac[h])
                    ac[h][l-1] = ac[h][l-1]/(pose[j+1][9]-pose[j][9])
                else:
                    ac[h].append(0)
                    


    ac_ = [[],[],[],[],[],[],[],[], []]
    count = 0
    i = 0
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    size = len(ac_[0])
    # obtain the acceleration for each 2d coordinates 
    while i <= 17:

            for j in range(len(ac[i])):
                ac_[count].append(m.sqrt(pow(ac[i][j],2)+pow(ac[i+1][j],2)))

            ac_[count] = np.array(ac_[count], dtype=float)    
            ac_[count] = smooth(ac_[count], round(size/2), 'bartlett')
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

    feature = [[],[],[],[],[]]
    # obtain the direction of segmented movement
    for i in range(len(seg)):
        if len(seg[i]):
            j=0
            joint = i
            while j < len(seg[i])-2:
                
                z1 = joint
                first_point_x = pose[seg[i][j]][z1].x

                if z1 == 4:
                    first_point_x_1 = first_point_x-pose[seg[i][j]][2].x
                    first_point_x_2 = first_point_x-pose[seg[i][j]][3].x 

                if z1 == 7:
                    first_point_x_1 = first_point_x-pose[seg[i][j]][5].x
                    first_point_x_2 = first_point_x-pose[seg[i][j]][6].x


                if z1 == 0:
                    first_point_x_1 = first_point_x-pose[seg[i][j]][8].x

                z2 = joint
                first_point_y = pose[seg[i][j]][z2].y
                
                if z2 == 4:
                    first_point_y_1 = first_point_y-pose[seg[i][j]][2].y
                    first_point_y_2 = first_point_y-pose[seg[i][j]][3].y 

                if z2 == 7:
                    first_point_y_1 = first_point_y-pose[seg[i][j]][5].y
                    first_point_y_2 = first_point_y-pose[seg[i][j]][6].y
                
                if z2 == 0:
                    first_point_y_1 = first_point_y-pose[seg[i][j]][8].y

                j = j+2
                second_point_x = pose[seg[i][j]][z1].x
                second_point_y = pose[seg[i][j]][z2].y

                if z1 == 4:
                    second_point_x_1 = second_point_x-pose[seg[i][j]][2].x
                    second_point_x_2 = second_point_x-pose[seg[i][j]][3].x 
                
                if z2 == 4:
                    second_point_y_1 = second_point_y-pose[seg[i][j]][2].y
                    second_point_y_2 = second_point_y-pose[seg[i][j]][3].y 
                
                if z1 == 7:
                    second_point_x_1 = second_point_x-pose[seg[i][j]][5].x
                    second_point_x_2 = second_point_x-pose[seg[i][j]][6].x 
                
                if z2 == 7:
                    second_point_y_1 = second_point_y-pose[seg[i][j]][5].y
                    second_point_y_2 = second_point_y-pose[seg[i][j]][6].y 

                if z1 == 0:
                    second_point_x_1 = second_point_x-pose[seg[i][j]][8].x
                
                if z2 == 0:
                    second_point_y_1 = second_point_y-pose[seg[i][j]][8].y

                if seg[i][j] in aux[i]:
                    end_point = 1
                else:
                    end_point = 0
                
                # conditions for bow gesture
                if z1 == 0:

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
                            if second_point_y_1 > first_point_y_1:
                                angle_lambda = 0
                            else:    
                                
                                angle_lambda = -(m.pi - m.acos((-second_point_y_1/first_point_y_1)))

                    else:
                    
                        if (second_point_y_1 <= 0):
                            angle_lambda = (m.acos(first_point_y_1/second_point_y_1))

                        else:
                            if second_point_y_1 < first_point_y_1:
                                angle_lambda = 0
                            else:

                                angle_lambda = (m.pi - m.acos((first_point_y_1*(-1))/second_point_y_1))

                    angle_lambda = round(m.degrees(angle_lambda),2)
                    feature[4].extend([angle_lambda])

                # conditions for handshake, hi and praying gestures
                if z1 == 4 or z1 == 7:
                    
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
                    
                    if z1 == 4:
                        feature[0].extend([angle])

                    if z1 == 7:
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
                    
                    if z1 == 4:
                        feature[2].extend([angle])

                    if z1 == 7:
                        feature[3].extend([angle])
                    
                
                j= j+end_point

    n = 0
    ml = 0

    for i in range(len(feature)):
        n = len(feature[i])
        if n > ml:
            ml = n

    for i in range(len(feature)):
        zeros = ml - len(feature[i])
        if zeros != 0:
            v = np.zeros(zeros)
            feature[i].extend(v)

    #print(feature)
    return feature




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



def quaternion_mult(q,r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def point_rotation_by_quaternion(point,q):
    r = [0]+point
    q_conj = [q[0],-1*q[1],-1*q[2],-1*q[3]]
    return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]

 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = m.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = m.atan2(R[2,1] , R[2,2])
        y = m.atan2(-R[2,0], sy)
        z = m.atan2(R[1,0], R[0,0])
    else :
        x = m.atan2(-R[1,2], R[1,1])
        y = m.atan2(-R[2,0], sy)
        z = 0
 
    return [x, y, z]

# Saves values for datasets
def DataSaving(x, y, d, h_angle) :
    
    data1['x'].append(x)
    data1['y'].append(y)
    data1['Distances'].append(d)
    print(x)
    print(y)
    print(d)

    key = getkey()
    if key == 'p':
        print('Saving dataset for NO eyegaze!')
            
        # write a csv file with the eyegaze features
        from pandas import DataFrame
        df = DataFrame(data1)
        df.to_csv ('distances12_no.csv', header=True) 

    elif key == 'q':
        print('Saving dataset for YES eyegaze!')
            
        # write a csv file with the eyegaze features
        from pandas import DataFrame
        df = DataFrame(data1)
        df.to_csv ('distances12_yes.csv', header=True) 

    elif key == 'r':
        print('Keep going!')

    elif key == 's':
        print('Looking head!')

        data2['x_angle'].append(h_angle[0])
        data2['y_angle'].append(h_angle[1])
        data2['z_angle'].append(h_angle[2])
        data2['classification'].append(1)

    elif key == 'd':
        print('No looking head!')

        data2['x_angle'].append(h_angle[0])
        data2['y_angle'].append(h_angle[1])
        data2['z_angle'].append(h_angle[2])
        data2['classification'].append(0)


    elif key == 'h':
        print('Saving dataset for headpose!')
            
        # write a csv file with the most headpose features
        from pandas import DataFrame
        df = DataFrame(data2)
        df.to_csv ('fit_headpose15.csv', header=True) 


# Detect if there is intention of interaction
def Intent_int():

    global d_pose
    global h_angle
    global d
    global Interaction
    global gestures
    global image
    global image2

    # find probability for the eyegaze
    if d > 1:
        d = 1

    if d <= 0.37:
        new_d = abs(d-0.37)
        new_d = new_d/0.37
        r = new_d*0.5
        eye_prob = 0.5+r

    else:
        new_d = 1-d
        new_d = new_d/0.63
        r = new_d*0.5
        eye_prob = 0.5-(0.5-r)

    # find the probability for headpose
    headpose_prediction = headpose_clf.predict_proba([h_angle])
    
    # find the probability for pose
    pose_prediction = pose_clf.predict_proba([[d_pose]])

    # calculate final probability of intention of interaction
    int_intent = eye_prob*0.47 + pose_prediction[0][1]*0.38 + headpose_prediction[0][1]*0.15
    #print(int_intent)
    
    if gestures != 1:
        if int_intent >= 0.45:
            print( 'Wants to interact!' )
            #image = cv2.imread('interaction.jpeg')

            Interaction = 1
        else:
            print( 'Do not wants to interact!' )
            #image = cv2.imread('interaction.jpeg')
            Interaction = 0

        #image = cv2.resize(image, (0, 0), None, .5, .5)
        #image2 = cv2.resize(image2, (0, 0), None, .5, .5)

        #numpy_horizontal_concat = np.concatenate((image, image2), axis=1)

        #cv2.imshow('image', numpy_horizontal_concat)
        #cv2.moveWindow('image', 2000,0) 


rospy.init_node('data_receiver')
openface = rospy.Subscriber('openface2/faces', Faces, openface_callback)
openpose = rospy.Subscriber('openpose_ros/human_list', OpenPoseHumanList, openpose_callback)
rospy.spin()
