#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataSet = pd.read_csv('Data.csv')
X=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,-1].values

np.set_printoptions(threshold = np.nan)

#taking care of missing data
#Imputer class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding categorical data 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])

oneHotEncoder_y = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_y.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y= labelEncoder_y.fit_transform(y)

#splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state= 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

