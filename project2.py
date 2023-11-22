# -*- coding: utf-8 -*-
"""project2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1328bRTZGCCm4cSogZpKQ8qsCzUR8-nV3

**Data Preprocessing**
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

!gdown 1VymOuQtYUfXBV4cdUrJusrx2kVA6cVGU
Adult_TrainDataset = pd.read_csv('Adult_TrainDataset.csv')
Adult_TrainDataset

!gdown 1X_ndzy4zZC_kCugo5cOobDFL_jY2WIy9
Adult_TestDataset = pd.read_csv('Adult_TestDataset.csv')
Adult_TestDataset

"""

---

"""

Adult_TrainDataset.drop('Capital_Gain', axis=1, inplace=True)
Adult_TrainDataset.drop('Capital_Loss', axis=1, inplace=True)
Adult_TrainDataset.replace({'?':np.NaN},  inplace=True)
Adult_TrainDataset

Adult_TestDataset.drop('Capital_Gain', axis=1, inplace=True)
Adult_TestDataset.drop('Capital_Loss', axis=1, inplace=True)
Adult_TestDataset.replace({'?':np.NaN},  inplace=True)
Adult_TestDataset

"""

---

"""

Adult_TrainDataset.info()

Adult_TestDataset.info()

"""

---

"""

Adult_TrainDataset.describe()

Adult_TestDataset.describe()

"""

---

"""

Adult_TrainDataset.isnull().sum()

Adult_TestDataset.isnull().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
Adult_TrainDataset['Native_Country'] = imputer.fit_transform(Adult_TrainDataset[['Native_Country']])
Adult_TrainDataset['Work_Class'] = imputer.fit_transform(Adult_TrainDataset[['Work_Class']])
Adult_TrainDataset['Occupation'] = imputer.fit_transform(Adult_TrainDataset[['Occupation']])

Adult_TrainDataset.isnull().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
Adult_TestDataset['Native_Country'] = imputer.fit_transform(Adult_TestDataset[['Native_Country']])
Adult_TestDataset['Work_Class'] = imputer.fit_transform(Adult_TestDataset[['Work_Class']])
Adult_TestDataset['Occupation'] = imputer.fit_transform(Adult_TestDataset[['Occupation']])

Adult_TestDataset.isnull().sum()

"""---

**Convert nominal value to numerical value**
1. One Hot Encoding
"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Adult_TrainDataset.Work_Class = le.fit_transform(Adult_TrainDataset.Work_Class)
Adult_TrainDataset.Marital_Status = le.fit_transform(Adult_TrainDataset.Marital_Status)
Adult_TrainDataset.Occupation = le.fit_transform(Adult_TrainDataset.Occupation)
Adult_TrainDataset.Relationship = le.fit_transform(Adult_TrainDataset.Relationship)
Adult_TrainDataset.Race = le.fit_transform(Adult_TrainDataset.Race)
Adult_TrainDataset.Sex = le.fit_transform(Adult_TrainDataset.Sex)
Adult_TrainDataset.Native_Country = le.fit_transform(Adult_TrainDataset.Native_Country)
Adult_TrainDataset.Income = le.fit_transform(Adult_TrainDataset.Income)

Adult_TrainDataset.info()

"""2. Label Encoding"""

from sklearn.preprocessing import LabelEncoder
categorical_column = ['Education']
encoder = LabelEncoder()
for column in categorical_column:
    Adult_TrainDataset[column] = encoder.fit_transform(Adult_TrainDataset[column])
Adult_TrainDataset.info()

from sklearn.preprocessing import LabelEncoder

let = LabelEncoder()

Adult_TestDataset.Work_Class = let.fit_transform(Adult_TestDataset.Work_Class)
Adult_TestDataset.Marital_Status = let.fit_transform(Adult_TestDataset.Marital_Status)
Adult_TestDataset.Occupation = let.fit_transform(Adult_TestDataset.Occupation)
Adult_TestDataset.Relationship = let.fit_transform(Adult_TestDataset.Relationship)
Adult_TestDataset.Race = let.fit_transform(Adult_TestDataset.Race)
Adult_TestDataset.Sex = let.fit_transform(Adult_TestDataset.Sex)
Adult_TestDataset.Native_Country = let.fit_transform(Adult_TestDataset.Native_Country)
Adult_TestDataset.Income = let.fit_transform(Adult_TestDataset.Income)

Adult_TestDataset.info()

from sklearn.preprocessing import LabelEncoder
categorical_columns2 = ['Education']
encoder2 = LabelEncoder()
for column2 in categorical_columns2:
    Adult_TestDataset[column2] = encoder2.fit_transform(Adult_TestDataset[column2])
Adult_TestDataset.info()

"""---

**Detect outliers**
"""

from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.05)
outlier_detector.fit(Adult_TrainDataset)
outlier_predictions = outlier_detector.predict(Adult_TrainDataset)
outlier_mask = outlier_detector.predict(Adult_TrainDataset) == -1
outliers = Adult_TrainDataset[outlier_mask]
print("Outliers:")
print(outliers)
num_outliers = len(outliers)
print("Number of outliers:", num_outliers)
Adult_TrainDataset = Adult_TrainDataset[~outlier_mask]
print("New Dataset without outliers:")
Adult_TrainDataset

from sklearn.ensemble import IsolationForest

outlier_detector2 = IsolationForest(contamination=0.05)
outlier_detector2.fit(Adult_TestDataset)
outlier_predictions2 = outlier_detector2.predict(Adult_TestDataset)
outlier_mask2 = outlier_detector2.predict(Adult_TestDataset) == -1
outliers2 = Adult_TestDataset[outlier_mask2]
print("Outliers:")
print(outliers2)
num_outliers2 = len(outliers2)
print("Number of outliers:", num_outliers2)
Adult_TestDataset = Adult_TestDataset[~outlier_mask2]
print("New Dataset without outliers:")
Adult_TestDataset

"""---

**1. Scatter Plot**
"""

import matplotlib.pyplot as plt

plt.scatter(Adult_TrainDataset['Sex'], Adult_TrainDataset['Age'], c=Adult_TrainDataset['Income'])

plt.xlabel('Sex')
plt.ylabel('Age')

cbar = plt.colorbar()
cbar.set_label('Income')

plt.show()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(Adult_TrainDataset['Age'], Adult_TrainDataset['Final_Weight'])

ax.set_xlabel('Age')
ax.set_ylabel('Final Weight')

ax.set_title('Scatter Plot of Age vs Final Weight')

plt.show()

"""**2.  Line Chart**"""

import matplotlib.pyplot as plt

x = Adult_TrainDataset['Hours-Per-Week']
y = Adult_TrainDataset['Income']

plt.plot(x, y)

plt.xlabel('Hours-Per-Week')
plt.ylabel('Income')
plt.title('Line Chart')

plt.show()

"""---

**Build and train models**

**Classification by method KNN**
"""

from sklearn.neighbors import KNeighborsClassifier

XTR = Adult_TrainDataset.drop(['Income','Age'], axis='columns')
YTR = Adult_TrainDataset.Income

XTE = Adult_TestDataset.drop(['Income','Age'], axis='columns')
YTE = Adult_TestDataset.Income

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(XTR, YTR)

trAcc = knn.score(XTR, YTR)
teAcc = knn.score(XTE, YTE)

print ('Train Accuracy: ', trAcc)
print ('Test Accuracy: ', teAcc)

import matplotlib.pyplot as plt

trAcc=[]
teAcc=[]
Ks=[]

for i in range(1,11):
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(XTR, YTR)
    trAcc.append(KNN.score(XTR, YTR))
    teAcc.append(KNN.score(XTE, YTE))
    Ks.append(i)

plt.plot(Ks, trAcc, label = 'Train')
plt.plot(Ks, teAcc, label = 'Test')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""**Confusion Matrix**"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = knn.predict(XTE)
cm = confusion_matrix(YTE, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(YTE, y_pred))

"""**Classification by method SVM**

"""

import sklearn.svm as sv

Clsfr = sv.SVC(kernel='rbf')
Clsfr.fit(XTR, YTR)

trAc = Clsfr.score(XTR, YTR)
teAc = Clsfr.score(XTE, YTE)

print ('Train Accuracy: ', trAc)
print ('Test Accuracy: ', teAc)

"""**Confusion Matrix**"""

Y_pred = Clsfr.predict(XTE)
cm = confusion_matrix(YTE, Y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(YTE, Y_pred))

"""**Decision Tree**"""

import sklearn.tree as tr

trAcc = []
teAcc = []
MD = []

for i in range(2, 12):
    DT = tr.DecisionTreeClassifier(max_depth = i)
    DT.fit(XTR, YTR)
    trAcc.append(DT.score(XTR, YTR))
    teAcc.append(DT.score(XTE, YTE))
    MD.append(i)

plt.plot(MD, trAcc, label = 'Train', marker = 'o')
plt.plot(MD, teAcc, label = 'Test', marker = 'o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

Y_pred0 = DT.predict(XTE)
cm = confusion_matrix(YTE, Y_pred0)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(YTE, Y_pred0))