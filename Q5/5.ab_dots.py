import numpy as np
import matplotlib.pyplot as plt

x =[]
y = []

with open("locations.csv") as f: 
	first_line = f.readline()	# skip first line
	for line in f:
		x.append(line.split(',')[1])
		y.append(line.split(',')[2])

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Positions of Tesla")	# figure title
ax1.set_xlabel('latitude')	# x axis label
ax1.set_ylabel('longitude')	# y axis label

ax1.plot(x, y, 'ro', label='Path of Tesla')	# line color=red, line label
legend = ax1.legend(loc='lower right', shadow=True)	# location of label

plt.show()