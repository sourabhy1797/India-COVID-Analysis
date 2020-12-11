import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline

#importing the dataset 
data4 = pd.read_csv(r'C:\Users\yadavs\Documents\coorelation india\Datasets\dataset3.csv')
corr = data4.corr()
corr

#data reshape
X = data4['New cases'].values.reshape(-1,1)
y = data4['Stringency Index'].values.reshape(-1,1)

#datasplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#regression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

#prediction
y_pred = regressor.predict(X_test)

#comparison of actual vs predicted
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

#error metric
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)