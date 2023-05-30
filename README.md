# DS-EXERCISE-10

# AIM :

 To Perform Data Science Process on a complex dataset and save the data to a file.
 
 # ALGORITHM:
 
 STEP 1: Read the given Data 
 
 STEP 2 Clean the Data Set using Data Cleaning Process 
 
 STEP 3 Apply Feature Generation/Feature Selection Techniques on the data set 
 
 STEP 4 Apply EDA /Data visualization techniques to all the features of the data set
 
 # CODE:
 
 ## Loading dataset
 
import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

df

df.head()

df.info()

df.tail()

df.isnull().sum()

df.shape

df.nunique()

## Feature scaling methods

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#### Example dataset

data = [[10, 20, 30],
       [5, 15, 25],
       [3, 6, 9],
       [8, 12, 16]]
       
#### Min-max scaling

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Min-max scaled data:")

print(scaled_data)

#### Standard scaling

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Standard scaled data:")

print(scaled_data)

## Data visualization

sns.catplot(x = 'Outcome Variable' , y = 'Age' , data = df , kind = "swarm")

plt.figure(figsize=(20,10))

sns.lineplot(x='Disease',y='Age',data =df);

plt.figure(figsize=(20,10))

sns.barplot(x='Fever',y='Age',data =df);

sns.displot(df['Age'] , kde=True)

df.groupby('Gender').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Cough' , kind='count',data=df , hue = "Cholesterol Level")

df.groupby('Blood Pressure').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Gender' , kind='count',data=df , hue = "Cholesterol Level")

df = df.iloc[:,1:]

df.groupby('Fatigue').size().plot(kind='pie', autopct='%.2f')

#### Histogram

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(8, 6))

plt.hist(data, bins=30, edgecolor='black')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()

## OUTPUT:

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-10/assets/126390051/3b77d019-281e-4c06-a6f9-e06b3942feb0)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-10/assets/126390051/e9b3d45d-8441-4bf1-9894-62e6187bfa57)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-10/assets/126390051/7aa3069e-a517-4d5e-8ac7-c210b566c5ad)


