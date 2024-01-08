#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U NLTK')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfTransformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

get_ipython().run_line_magic('matplotlib', 'inline')

# read training data
train_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/data/cornell_movie/train.csv"
train_df = pd.read_csv(train_file, header=0)
print(f'num train records: {len(train_df)}')

# read test data
test_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/data/cornell_movie/test.csv"
test_df = pd.read_csv(test_file, header=0)
print(f'num test records: {len(test_df)}')


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
# train data
text_train = train_df['text']
y_train = train_df['y_label']

# test data
text_test = test_df['text']
y_test = test_df['y_label']

tfidfVec = TfidfVectorizer(min_df=5,
                           tokenizer=nltk.word_tokenize,
                           max_features=3000)

X_train = tfidfVec.fit_transform(text_train)
X_test = tfidfVec.transform(text_test)


# In[3]:


#2a

from sklearn import tree
from sklearn.metrics import accuracy_score

maxdepths = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
for depth in maxdepths:
  clf_tree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=depth)
  clf_tree.fit(X_train, y_train)
  y_predict = clf_tree.predict(X_test)
  print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_predict)))

#A max depth of 6 has the highest accuracy score for this decision tree.


# In[4]:


#2b

import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

reguList = [0.1, 0.5, 1.0,  5, 10, 20, 50, 100]
for regu in reguList:
  clf_LR = LogisticRegression(penalty='l2', C=regu,fit_intercept=True)
  clf = sklearn.linear_model.LogisticRegression(fit_intercept=True)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_pred)))

#The different inputs from the given list all have the same accuracy score of 80%.


# In[5]:


#2c

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

reguList = [0.1, 0.5, 1.0,  5, 10, 20, 50, 100]
for regu in reguList:
  clf_svm = SVC(C=regu, kernel='rbf')

  clf.fit(X_train, y_train)

  Y_pred = clf.predict(X_test)

  print('Accuracy on test data is %.2f' % (accuracy_score(y_test, Y_pred)))

#The different inputs from the given list all have the same accuracy score of 80%.


# In[ ]:


# The best model is the Logistic Regression model because it uses a penalty feature that reduces the complexity of the model and improves the generalization by reducing the variance of the model.


# In[6]:


import pandas as pd

# read training data
train_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_train.csv"
train_df = pd.read_csv(train_file, header='infer')


# In[7]:


# read test data
test_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_test.csv"
test_df = pd.read_csv(test_file, header='infer')


# In[8]:


cols = train_df.columns
cols


# In[9]:


x_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[10]:


# train data
X_train = train_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y_train = train_df['Outcome'].values

# test data
X_test = test_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y_test = test_df['Outcome'].values


# In[11]:


#3a
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)

clf_tree.fit(X_train, y_train)

y_predict = clf_tree.predict(X_test)

print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_predict)))
print('Precision Score on test data is %.2f' % (precision_score(y_test, y_predict)))
print('Recall Score on test data is %.2f' % (recall_score(y_test, y_predict)))
print('F1 Score on test data is %.2f' % (f1_score(y_test, y_predict)))


# In[12]:


#3b
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier( DecisionTreeClassifier(criterion='entropy', max_depth=4),
                            n_estimators=500,
                            max_samples=100,
                            bootstrap=True,
                            random_state=42)

bag_clf.fit(X_train, y_train)

y_predict = bag_clf.predict(X_test)

print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_predict)))
print('Precision Score on test data is %.2f' % (precision_score(y_test, y_predict)))
print('Recall Score on test data is %.2f' % (recall_score(y_test, y_predict)))
print('F1 Score on test data is %.2f' % (f1_score(y_test, y_predict)))


# In[13]:


#3c
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=4, random_state=42)

rnd_clf.fit(X_train, y_train)

y_predict = rnd_clf.predict(X_test)

print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_predict)))
print('Precision Score on test data is %.2f' % (precision_score(y_test, y_predict)))
print('Recall Score on test data is %.2f' % (recall_score(y_test, y_predict)))
print('F1 Score on test data is %.2f' % (f1_score(y_test, y_predict)))


# In[14]:


#3d
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=500,algorithm="SAMME.R", learning_rate=0.5, random_state=42)

ada_clf.fit(X_train, y_train)

y_predict = ada_clf.predict(X_test)

print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_predict)))
print('Precision Score on test data is %.2f' % (precision_score(y_test, y_predict)))
print('Recall Score on test data is %.2f' % (recall_score(y_test, y_predict)))
print('F1 Score on test data is %.2f' % (f1_score(y_test, y_predict)))


# In[ ]:


#The ensemble methods were able to improve the base model by combining the predictions of several base estimators built with a given learning algorithm in order to improve the generalizability and robustness over a single estimator.
#For Bagging, it forms a class of algorithms which build several instances of a black-box estimator on random subsets of the original training set and then aggregates their individual predictions to form a final prediction.
#For RandomForest, each tree in the ensemble is built from a sample drawn with replacement from the training set. Then, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of the max features.
#For AdaBoost, it fits a sequence of weak learners on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted sum to produce the final prediction.

