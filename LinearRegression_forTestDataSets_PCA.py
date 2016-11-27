import pandas as pd
import numpy as np
from sklearn import linear_model
import os
from itertools import izip, count

TRAINING_DATASET = 'Training_PCA.csv'
TRAINING_Y = 'TrainY.csv'
TEST_DATASET_DIRECTORY = 'testPCA'	# directory name that contains test datasets

# Read training dataset
df = pd.read_csv(TRAINING_DATASET, header=None)		# read from the first line
columns = len(df.columns)
rows = len(df.index)
print 'Training dataset:', "{:,}".format(len(df.index)), 'x', "{:,}".format(len(df.columns))
df_x = df.ix[:,:]
X = np.array(df_x)

# Read training Y
df = pd.read_csv(TRAINING_Y, header=None)		# read from the first line
print 'Training dataset:', "{:,}".format(len(df.index)), 'x', "{:,}".format(len(df.columns))
df_y = df.ix[:,:]
Y = np.array(df_y)

reg = linear_model.LinearRegression()
reg.fit(X,Y)
coeffs = reg.coef_
print 'coeffs', coeffs

# Read Test dataset
testFiles = [file for file in os.listdir(TEST_DATASET_DIRECTORY) if str(file).find('test') >= 0]
print 'Number of test files:', len(testFiles)

TEST_Y_ALL = np.array([])
TEST_Y_ALL_PREDICTED = np.array([])
for file in testFiles:
	df = pd.read_csv(TEST_DATASET_DIRECTORY + '/' + file, header=None)		# read from the first line
	df_y = df.ix[:,columns]
	df_x = df.ix[:,:columns-1]
	
	X = np.array(df_x)
	Y = np.array(df_y)
	
	X = X.transpose()
	predictedY = np.dot(coeffs, X) + reg.intercept_
	
	TEST_Y_ALL = np.append(TEST_Y_ALL, Y)
	TEST_Y_ALL_PREDICTED = np.append(TEST_Y_ALL_PREDICTED, predictedY)
		
print 'TEST_Y_ALL size:', "{:,}".format(len(TEST_Y_ALL))
print 'TEST_Y_ALL_PREDICTED size:', "{:,}".format(len(TEST_Y_ALL_PREDICTED))

TEST_Y_ALL_PREDICTED_BELOW_ZERO_REPLACED_TO_ZERO = []
for i in range(len(TEST_Y_ALL_PREDICTED)):
	if TEST_Y_ALL_PREDICTED[i] < 0:
		TEST_Y_ALL_PREDICTED_BELOW_ZERO_REPLACED_TO_ZERO.append(0)
	else:
		TEST_Y_ALL_PREDICTED_BELOW_ZERO_REPLACED_TO_ZERO.append(TEST_Y_ALL_PREDICTED[i])

COMPARE = np.vstack([TEST_Y_ALL, TEST_Y_ALL_PREDICTED_BELOW_ZERO_REPLACED_TO_ZERO])
COMPARE = COMPARE.transpose()
print COMPARE
dfCOMPARE = pd.DataFrame(COMPARE)
dfCOMPARE.to_csv('Compare_LR_PCA.csv', sep=',')

ERROR = abs(TEST_Y_ALL - TEST_Y_ALL_PREDICTED_BELOW_ZERO_REPLACED_TO_ZERO)
print 'Linear Regression for PCA'
print 'Mean error:', ERROR.mean()