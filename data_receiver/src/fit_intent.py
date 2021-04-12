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
train1=pd.read_csv("datasets(intent_interaction/intent_yes3.csv", header=0, converters={"Intent_prob":float})
train2=pd.read_csv("datasets(intent_interaction/intent_no3.csv", header=0, converters={"Intent_prob":float})

# obtain the intent probability
intent1 = train1['Intent_prob']
intent2 = train2['Intent_prob']
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
n_look, bins1, p1 = hist(intent1,normed=0, range=(0,1), bins = 20, label='potential intent points')
n_nolook, bins2, p2 = hist(intent2,normed=0, range=(0,1), bins = 20, color='red', label='no potential intent points')
plt.xlabel('Probabilities of the potential intent detector')
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
    prob_false_det = 0
    while m < i:
        prob_false_det = prob_false_det + (n_look[m]/total_looking)
        m = m+1

    m = len(n_look)-1
    prob_fail_det = 0
    while m > i:
        prob_fail_det = prob_fail_det + (n_nolook[m]/total_nolooking)
        m = m-1

    final_error = prob_fail_det+prob_false_det
    Probabilitie_error.append(final_error)   

print(Probabilitie_error)

# Get the best T for probabilite error
i = Probabilitie_error.index(min(Probabilitie_error))
t = T[i]

print(t)
