import csv
import pandas as pd
import numpy as np

from sklearn import linear_model

df_x = pd.read_csv("Training_ZScore.csv")
df_y = pd.read_csv("TrainY.csv")

X = np.array(df_x)
Y = np.array(df_y)
Y = np.ravel(Y)

min_diff = 99999
min_alpha = 0

alphavalues = np.arange(2,10,0.1)

for i in alphavalues:
	lasso = linear_model.Lasso(alpha=i, fit_intercept=True, selection='random')
	lasso.fit(X,Y)
	ysame = lasso.predict(X)
	mean_diff = abs(Y-ysame).mean()

	print str(i)+' '+str(mean_diff)

	if(mean_diff< min_diff):
		min_diff = mean_diff
		min_alpha = i

print min_alpha
print min_diff

