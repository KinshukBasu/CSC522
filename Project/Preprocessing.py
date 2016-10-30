# This program is for preprocessing Blog Post training data set.
# It is not completed yet. I made it just as the first step.
# See details below.
# Read input_filename and write output_filename
# Calculate Z-score for Attribute 51-55, and write
# Calculate Z-score for Attribute 56-60, and write
# Other Attribute are just appended

import csv

everyrow = []	
input_filename = 'blogData_train.csv'
output_filename = 'blogData_Preprocessed.csv'

with open(input_filename, 'rb') as f1:
	reader = csv.reader(f1)
	for row in reader:	
		elements = []
		
		# Attribute 51-55
		# Z-score is calculated, and written
		# Z-score of 51 = (51-1)/2
		# index is decreased by 1, so (50-0)/1
		base = 50
		mul = 5
		for i in range(0, 5):
			value = float(row[base+i])	# each value, x
			mean = float(row[mul*i])	# mean, x bar
			standardDeviation = float(row[mul*i+1])		# standard deviation
			if(standardDeviation != 0.0):
				elements.append((value-mean)/standardDeviation)	# Z-score
			else:
				elements.append(value-mean)		# when standard deviation is zero, no division
		
		# Attribute 56-60
		# Z-score is calculated, and written
		# Z-score of 56 = (56-26)/27
		# index is decreased by 1, so (55-25)/26
		base = 55
		mulbase = 25
		for i in range(0, 5):
			value = float(row[base+i])	# each value, x
			mean = float(row[mul*i+mulbase])	# mean, x bar
			standardDeviation = float(row[mul*i+1+mulbase])		# standard deviation
			if(standardDeviation != 0.0):
				elements.append((value-mean)/standardDeviation)	# Z-score
			else:
				elements.append(value-mean)		# when standard deviation is zero, no division
		
		# Attribute 61-281
		# index is decreased by 1, 60-280
		# just appended
		for i in range(60, 281):
			elements.append(row[i]);
		
		# Append each row
		everyrow.append(elements)	# append each row
		
with open(output_filename, 'wb') as f2:
	mywriter = csv.writer(f2)
	for row in everyrow:	
		mywriter.writerow(row)	# write to a new file