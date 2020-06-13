#!/usr/bin/env python
# coding: utf-8

# In[28]:


#IMPORING PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy.stats import expon,uniform,randint
Employee_Attrition=pd.read_csv(r'C:\\PythonD\\lib\\EmployeeAttrition.csv')
Employee_Attrition


# In[29]:


#Dimensions of Datase
Employee_Attrition.shape


# In[30]:


#Inspection of Dataset
Employee_Attrition.dtypes


# In[45]:


#Checking for null values
Employee_Attrition.isnull()


# In[32]:


Employee_Attrition.isnull().sum(axis=0)


# In[33]:


Employee_Attrition.head()


# In[34]:


Employee_Attrition.tail()


# In[35]:


labels=Employee_Attrition['BusinessTravel']
Employee_Attrition.drop(['BusinessTravel','Department','Education','EducationField','EmployeeCount','EmployeeNumber','StandardHours'],axis=1,inplace=True)


# In[36]:


Employee_Attrition.head()


# In[37]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split,RandomizedSearchCV,cross_val_score,cross_val_predict,validation_curve
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# # TRAIN-TEST SPLIT

# In[39]:


y=Employee_Attrition['Attrition']
X=Employee_Attrition[Employee_Attrition.columns.difference(['Attrition'])]


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=99, stratify=y)


# In[41]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# In[42]:


print(X_train.shape)
print(X_test.shape)


# # Normalization

# In[43]:


from sklearn.preprocessing import MinMaxScaler
full_scaler = MinMaxScaler()
full_scaler.fit(X_train)
X_train = pd.DataFrame(full_scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(full_scaler.transform(X_test), columns=X_test.columns)


# # Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logistic = LogisticRegression(class_weight='balanced')
logistic.fit(X_train, y_train)

logistic_train_preds = logistic.predict(X_train)
logistic_test_preds = logistic.predict(X_test)

print(confusion_matrix(y_train, logistic_train_preds))
print(confusion_matrix(y_test, logistic_test_preds))

print(classification_report(y_train, logistic_train_preds))
print(classification_report(y_test, logistic_test_preds))


# In[ ]:




