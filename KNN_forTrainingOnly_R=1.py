# KNN
# RadiusNeighborsRegressor

import pandas as pd
from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np

KNN_RADIUS = 1.0

df = pd.read_csv("blogData_train.csv", header=None)		# read from the first line

df_all_x = df.ix[:,0:279]
df_all_y = df.ix[:,280]
row_count = len(df.index)

predictedY = []

for i in range(row_count):
	df_removed_one_x = df_all_x
	df_removed_one_x = df_removed_one_x.drop(df_removed_one_x.index[[i]])
	X = np.array(df_removed_one_x)
	#print('X')
	#print(X)
	
	df_x_test = df_all_x.iloc[i]
	X_TEST = np.array(df_x_test)
	#print('X_TEST')
	#print(X_TEST)
	
	df_removed_one_y = df_all_y
	df_removed_one_y = df_removed_one_y.drop(df_removed_one_y.index[[i]])
	Y = np.array(df_removed_one_y)
	#print('Y')
	#print(Y)
	
	neigh = RadiusNeighborsRegressor(radius = KNN_RADIUS)
	neigh.fit(X, Y)
	
	predicted_one_y = neigh.predict([X_TEST])
	predicted_one_y_2 = float(np.asarray(predicted_one_y))
	predictedY.append(predicted_one_y_2)
	
	print(repr(i+1) + ' / ' + repr(row_count))
#print(predictedY)
np.savetxt("predicted_Y_KNN_RADIUS_1.csv", predictedY, delimiter=",", fmt='%10.10f')
	
	
