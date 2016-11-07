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

df = pd.read_csv("PCAd_data.csv")
df_x = df

X = np.array(df_x)
Y = np.array(df_y)


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print (X_poly.shape)
reg = linear_model.LinearRegression()
reg.fit(X_poly, Y)
predictedY = reg.predict(X_poly)

yresid = Y-predictedY
plt.plot(yresid, 'x')
plt.show()

SSresid = sum(pow(yresid,2))
SStotal = len(Y)*np.var(Y)
R2 = 1 - SSresid/SStotal
print R2