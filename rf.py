# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:36:07 2021

@author: Sindhu D
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,classification_report,confusion_matrix,roc_curve, roc_auc_score,auc
import sklearn.metrics as metrics
import pickle

data = pd.read_csv("model.csv")

print("Data Head\n",data.head())

print("Data Describe\n", data.describe())

print("Data Shape\n",data.shape)

print("Data Label Count\n\n", data["Label"].value_counts())

# Data is imbalance

plt.hist(data.Label)
plt.title("Checking Data is balanced or Not, using hist",fontweight ="bold")
plt.show()

# CountVectorizer

#### The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary. You can use it as follows: Create an instance of the CountVectorizer class

count_vect = CountVectorizer(max_features = 3000)
x = count_vect.fit_transform(data['Review'])
pickle.dump(count_vect, open('cv.pkl', 'wb'))

print("count_vector shape\n\n",x.shape)

# # SMOTE - Synthetic Minority Oversampling Technique

##### way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model.

over_sample = SMOTE(random_state = 42, sampling_strategy = "all")

X_train_oversample, y_train_oversample = over_sample.fit_sample(x,data['Label'])

print("y_Train over sample \n\n", y_train_oversample.value_counts())

print("x_train over sample", X_train_oversample.shape)

colors = ['lime'] 
  
plt.hist(y_train_oversample, 
         density = True,  
         histtype ='barstacked', 
         color = colors)  
  
plt.title('balanced Data\n\n', 
          fontweight ="bold") 
  
plt.show()

x_train,x_test,y_train,y_test = train_test_split(X_train_oversample, y_train_oversample, test_size = 0.3, random_state = 42)

print("X_train shape=",x_train.shape,'\n',"y_train shape=",y_train.shape)
print("x_test shape=",x_test.shape,'\n',"Y_test shape=" ,y_test.shape)


# Random_Forest_Classifiers

RF = RandomForestClassifier(n_estimators = 120,
                           random_state = 50,
                           n_jobs = -1,
                           max_features = 'auto')
RF.fit(X_train_oversample,y_train_oversample)

rf_pred = RF.predict(x_test)

print("Recall=",recall_score(y_test, rf_pred, average='micro'),'\n')
print("Accuracy of Random forest classifier=", accuracy_score(y_test,rf_pred),'\n')
print("Classification Report:\n", classification_report(y_test,rf_pred),'\n')
print("Confusion Matrix \n", confusion_matrix(y_test,rf_pred),'\n')

