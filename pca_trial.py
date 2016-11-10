import csv
import math
import pandas as pd
import jupyter as jpy
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import curve_fit


df = pd.read_csv("blogData_train.csv")

df_y = df.ix[:,280]

df = pd.read_csv("TransX_8PC.csv")
df_x = df

X = np.array(df_x)
Y = np.array(df_y)

#----------PLAIN LINEAR REGRESSION----------------
"""
reg = linear_model.LinearRegression()
reg.fit(X,Y)
coeffs= reg.coef_

predictedY = np.dot(X,coeffs) + reg.intercept_

error = pow((Y-predictedY),2)
print "RMSE error in prediction : ",
print math.sqrt((error.sum())/len(Y))


plt.plot(Y, 'D')
plt.plot(predictedY,'s')
plt.show()

yresid = Y-predictedY
plt.plot(yresid, 'x')
plt.show()

SSresid = sum(pow(yresid,2))
SStotal = len(Y)*np.var(Y)
R2 = 1 - SSresid/SStotal
print R2

with open('linear_results.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(predictedY)
    wr.writerow(yresid)

#-----------POLYNOMIAL REGRESSION-------------
"""
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print (X_poly.shape)
reg = linear_model.LinearRegression()
reg.fit(X_poly, Y)
predictedY = reg.predict(X_poly)

error = pow((Y-predictedY),2)
print "RMSE error in prediction : ",
print math.sqrt((error.sum())/len(Y))

yresid = Y-predictedY
plt.plot(yresid, 'x')
plt.show()

SSresid = sum(pow(yresid,2))
SStotal = len(Y)*np.var(Y)
R2 = 1 - SSresid/SStotal
print R2
