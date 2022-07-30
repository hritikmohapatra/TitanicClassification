# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 23:35:46 2021

@author: HRITIK
"""

#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Data Collection & Processing
data = pd.read_csv("titanic.csv")

# number of rows and Columns
data.shape

data.info()

# check the number of missing values in each column
data.isnull().sum()

#Handling the Missing values
data = data.drop(columns='Cabin', axis=1)
data["Age"].fillna(data["Age"].mean(),inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data.isnull().sum()

#Data Analysis
data.describe()

data['Survived'].value_counts()

#Data Visualization
# making a count plot for "Survived" column
sns.countplot('Survived', data=data)

# making a count plot for "Sex" column
sns.countplot('Sex', data=data)

# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=data)

#Encoding the Categorical Columns
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

le1 = LabelEncoder()
data["Embarked"] = le1.fit_transform(data["Embarked"])

#Separating features & Target
x = data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
y = data['Survived']

#Splitting the data into training data & Test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=30, shuffle=True)

#Model Training
model = LogisticRegression()
model.fit(x_train, y_train)

#Model Evaluation
#model performance on training data
training_accuracy = model.score(x_train, y_train)
print('Accuracy score of training data : ', training_accuracy)

#prediction on test data
y_pred = model.predict(x_test)

#model performance on test data
test_accuracy = model.score(x_test, y_test)
print('Accuracy score of test data : ', test_accuracy)

#Accuracy Score
accuracy_score(y_test,y_pred)

#confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

