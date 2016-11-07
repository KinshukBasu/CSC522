import csv
import pandas as pd
import jupyter as jpy
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


from scipy.optimize import curve_fit


df = pd.read_csv("blogData_train.csv")

df_y = df.ix[:,280]
df_x = df.ix[:,0:279]


X = np.array(df_x)
Y = np.array(df_y)


reg = linear_model.LinearRegression()
reg.fit(X,Y)
coeffs= reg.coef_

print X.shape
print coeffs.shape
predictedY = np.dot(X,coeffs) + reg.intercept_
print predictedY.shape

error = abs(Y-predictedY)
print "Mean error in prediction : ",
print error.mean()


plt.plot(Y)
plt.plot(predictedY)
plt.show()

yresid = Y-predictedY
plt.plot(yresid, 'x')
plt.show()

SSresid = sum(pow(yresid,2))
SStotal = len(Y)*np.var(Y)
R2 = 1 - SSresid/SStotal
print R2


#Experimentation with polynomial regression
#http://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn
#http://stats.stackexchange.com/questions/70712/python-creating-a-polynomial-model-with-two-input-variables

"""
df_x = df.ix[:,0:60]
X = np.array(df_x)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print (X_poly.shape)
reg = linear_model.LinearRegression()
reg.fit(X_poly, Y)
predictedY = reg.predict(X_poly)


#plt.plot(Y)
#plt.plot(predictedY)
#plt.show()


yresid = Y-predictedY
plt.plot(yresid, 'x')
plt.show()

SSresid = sum(pow(yresid,2))
SStotal = len(Y)*np.var(Y)
R2 = 1 - SSresid/SStotal
print R2
"""
#[p,v] = np.polyfit(X,Y,5);
