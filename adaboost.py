import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import AdaBoostRegressor


df_x = pd.read_csv("Training_ZScore.csv")
df_y = pd.read_csv("TrainY.csv")
#test1 = pd.read_csv()

X = np.array(df_x)
Y = np.array(df_y)
Y = np.ravel(Y)

losstype = ['linear','square','exponential',]

min_mean = 999999
minloss = ''
min_n = 0

data_list = []


for loss in losstype:
	for n in range(100,5000,100):

		ada_1 = AdaBoostRegressor(n_estimators = n,loss= loss)
		ada_1.fit(X,Y)
		ysame = ada_1.predict(X)
		mean_diff = abs(Y-ysame).mean()

		print loss +' '+str(n)+' '+str(mean_diff)
		data_list.append([loss, n, mean_diff])

		if(mean_diff<min_mean):
			min_mean = mean_diff
			minloss = loss
			min_n = n


with open("ada_data.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(data_list)


print min_mean
print minloss
print min_n

"""
6.91684070994
square
700
"""