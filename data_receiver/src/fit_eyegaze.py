# import required libraries
from scipy import stats  
from scipy.stats import norm,rayleigh
from numpy import linspace
from pylab import plot,show,hist,figure,title
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import pickle

# disable a panda warning
pd.options.mode.chained_assignment = None  # default='warn'

# read the distances from the csv
train1=pd.read_csv("datasets/eye_gaze_distances/distances12_yes.csv", header=0, converters={"Distances":float, "x":float, "y":float})
train2=pd.read_csv("datasets/eye_gaze_distances/distances12_no.csv", header=0, converters={"Distances":float, "x":float, "y":float})

# obtain the x and y distance of the eye gaze
plt.figure(1)
title('Distribution of the looking points in 2D')
plt.xlabel('Distance along x [meters]')
plt.ylabel('Distance along y [meters]')
x = train1['x']
y = train1['y']
plt.scatter(x, y)
plt.gca().invert_yaxis()
plt.show()

plt.figure(2)
title('Distribution of the no looking points in 2D')
plt.xlabel('Distance along x [meters]')
plt.ylabel('Distance along y [meters]')
x = train2['x']
y = train2['y']
plt.scatter(x, y)
plt.gca().invert_yaxis()
plt.show()


# obtain the distance from the end of eye gaze vector and the center of the camera
distances1 = train1['Distances']
distances2 = train2['Distances']
#param = rayleigh.fit(distances) # distribution fitting

#x = linspace(0,1,100)
# fitted pdf distribution
#pdf_fitted = rayleigh.pdf(x,loc=param[0],scale=param[1])
# reverse array
#x_ = x[::-1]
# fitted cdf distribution
#cdf_fitted = rayleigh.cdf(x_,loc=param[0],scale=param[1])

# plot stuff
fig , ax = plt.subplots()
n_look, bins1, p1 = hist(distances1,normed=0, range=(0,1), bins = 100, label='looking points')
n_nolook, bins2, p2 = hist(distances2,normed=0, range=(0,1), bins = 100, color='red', label='no looking points')
plt.xlabel('Distances of the eyegaze points [meters]')
plt.ylabel('Frequency')
title('Histogram')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()

#fig , ax = plt.subplots()
#ax.plot(x, pdf_fitted, 'r-', label='pdf')
#ax.plot(x, cdf_fitted, 'b-', label='cdf')
#title('Rayleigh distribution')
#plt.xlabel('Distances of the looking points')
#plt.ylabel('Frequency')


total_looking = sum(n_look)
total_nolooking = sum(n_nolook)
print(total_looking)
print(total_nolooking)

# find the best T for minimize error probabilitie
Probabilitie_error = []
T = []
for i in range(1, len(n_look)):

    T.append(bins1[i])
    
    m=0
    prob_fail_det = 0
    while m < i:
        prob_fail_det = prob_fail_det + (n_nolook[m]/total_nolooking)
        m = m+1

    m = len(n_look)-1
    prob_false_det = 0
    while m > i:
        prob_false_det = prob_false_det + (n_look[m]/total_looking)
        m = m-1

    final_error = prob_fail_det+prob_false_det
    Probabilitie_error.append(final_error)   

print(Probabilitie_error)

# Get the best T for probabilite error
i = Probabilitie_error.index(min(Probabilitie_error))
t = T[i]

print(t)
