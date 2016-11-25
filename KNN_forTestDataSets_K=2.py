import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import os
from itertools import izip, count

TRAINING_DATASET = 'blogData_train.csv'
TEST_DATASET_DIRECTORY = 'Blog test'	# directory name that contains test datasets

KNN_K_VALUE = 2

# Read training dataset
df = pd.read_csv(TRAINING_DATASET, header=None)		# read from the first line

columns = len(df.columns)
rows = len(df.index)

print 'Training dataset:', "{:,}".format(len(df.index)), 'x', "{:,}".format(len(df.columns))

df_y = df.ix[:,columns-1]
df_x = df.ix[:,:columns-2]

X = np.array(df_x)
Y = np.array(df_y)

neigh = KNeighborsRegressor(n_neighbors = KNN_K_VALUE)
neigh.fit(X, Y)

# Read Test dataset
testFiles = [file for file in os.listdir(TEST_DATASET_DIRECTORY) if str(file).find('test') >= 0]
print 'Number of test files:', len(testFiles)

TEST_Y_ALL = np.array([])
TEST_Y_ALL_PREDICTED = np.array([])
for file in testFiles:
	df = pd.read_csv(TEST_DATASET_DIRECTORY + '/' + file, header=None)		# read from the first line
	df_y = df.ix[:,columns-1]
	df_x = df.ix[:,:columns-2]
	
	X = np.array(df_x)
	Y = np.array(df_y)
	
	predictedY = neigh.predict(X)
	
	TEST_Y_ALL = np.append(TEST_Y_ALL, Y)
	TEST_Y_ALL_PREDICTED = np.append(TEST_Y_ALL_PREDICTED, predictedY)
		
print 'TEST_Y_ALL size:', "{:,}".format(len(TEST_Y_ALL))
print 'TEST_Y_ALL_PREDICTED size:', "{:,}".format(len(TEST_Y_ALL_PREDICTED))

ERROR = abs(TEST_Y_ALL - TEST_Y_ALL_PREDICTED)
print 'Method: KNN for K=', KNN_K_VALUE
print 'Mean error:', ERROR.mean()

