# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KISHORE B
RegisterNumber: 212223240073
*/
```
```python
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
To Read Head and Tail Files

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/0f505b88-07a2-4c89-ac07-aad9d2a6a80b)


Compare Dataset

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/ab4e1866-d933-4a04-b4e9-696b83a6a83f)


Predicted Value

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/4eea1d85-36b6-4471-9cfd-6c692cf00a4e)


Graph For Training Set

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/007d6f64-9451-4822-9d8b-3d0e755b3d41)


Graph For Testing Set

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/c7d477a8-1d6b-4df1-b975-31829892f69a)


Error

![image](https://github.com/codedbykishore/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139122/af14bc9d-1ef9-473f-bba2-a050a9f9f74e)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
