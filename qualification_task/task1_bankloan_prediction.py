# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
import openpyxl 

# Load the dataset
file_path = 'bank_loan_data.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(df.head())

# number of rows and columns
df.shape

df.describe()

# number of missing values in each columns

df.isnull().sum()

# dropping the missing values
df = df.dropna()
df.isnull().sum()

df['Family'].value_counts()

#Data Visualization

df['Gender'].value_counts()

# replacing the values of - and # to O
df = df.replace(to_replace=('#','-'),value ='O')
df['Gender'].value_counts()

# convert categorical cols to numerical values
df.head()
df = df.replace({'Gender':{'M':0,'F':1,'O':2}})

df = df.replace({'Home Ownership':{'Home Mortage':0,'Home Owner':1,'Rent':3}})
df = df.replace({'Personal Loan':{' ':0}})

df.head()
#Education and Loan Status

sns.countplot(x="Education", hue = "Personal Loan", data = df)

# Gender and Loan Status

sns.countplot(x="Gender", hue = "Personal Loan", data = df)

# seperating the data and label

df['ZIP Code'].value_counts()
X = df.drop(columns = ['ID', 'Personal Loan', 'ZIP Code'], axis = 1)
Y = df['Personal Loan']

print(X)
print(Y)

ages = df['Personal Loan'].tolist()

print(ages)
df['Personal Loan'].value_counts()
# Train Test Split

# Create 20% test data, by including diverse data types in test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Training the model
# Support Vector Machine Model

classifier = svm.SVC(kernel= 'linear')

# training the support Vector Machine model
classifier.fit(X_train, Y_train)

# Model Evaluation

# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('accuracy on training data using svm: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy on test data using svm: ', test_data_accuracy)

# Making a predictive system

print(X_test.iloc[1])
X_new = X_train.iloc[1]

# Reshape the row: converts the row to a NumPy
X_new = X_new.values.reshape(1, -1)
prediction = classifier.predict(X_new)
print(prediction)

if(prediction == 0):
  print("Customer denied the Personal Loan")
else:
  print("Customer accepted the Personal Loan")
  
print(Y_test.iloc[1])

# training data using Gradient Boost
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=2)
gradient_boosting_model.fit(X_train, Y_train)

# Making prediction
X_train_prediction_gradBoost = gradient_boosting_model.predict(X_test)

# accuracy score on test data
test_data_accuracy_gradBoost = accuracy_score(X_train_prediction_gradBoost, Y_test)
print('accuracy on test data using gradient boost: ', test_data_accuracy_gradBoost)
