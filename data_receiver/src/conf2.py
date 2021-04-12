import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


from sklearn.metrics import confusion_matrix
import seaborn as sns

# read the data
train1=pd.read_csv("datasets/intent_interaction/interaction_label.csv", header=0, converters={"real":int})
train2=pd.read_csv("datasets/intent_interaction/interaction.csv", header=0, converters={"interaction":int}) 

# obtain the features and classification
pred = train2['interaction']
real = train1['real']

class_names = ['no interaction', 'interaction']

target = list(real)
y_pred = list(pred)

cm = confusion_matrix(target, y_pred)

# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=2)#for label size
sns.heatmap(cmn, annot=True, annot_kws={"size": 25}, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Interaction')
plt.xlabel('Predicted Interaction')
plt.show() 
