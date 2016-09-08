import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kn
from math import*
from decimal import Decimal
from scipy.spatial import distance
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pyplot as plt

x = []
y = []

with open("locations.csv") as f: 
	first_line = f.readline()	# skip first line
	for line in f:
		x.append(line.split(',')[1])
		y.append(line.split(',')[2])

xf = [float(xi) for xi in x]
yf = [float(yi) for yi in y]
		
lat_mean = np.average(xf)
long_mean = np.average(yf)

print 'Mean of latitude values: %f' % (lat_mean)
print 'Mean of longitude values: %f' % (long_mean)

xp = [lat_mean, long_mean]

# 4. Minkowski Distance (General)
def minkowski_distance(x, y, p_value):
	return pow(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)), 1./p_value)
	
Minkowski_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Minkowski_d.append(minkowski_distance(xm, xp, 3))

# 1. Euclidean distance
Euclidean_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Euclidean_d.append(minkowski_distance(xm, xp, 2))
	
# 2. Mahalanobis distance
Mahalanobis_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	VI = np.linalg.inv(np.cov(xf, yf))
	Mahalanobis_d.append(distance.mahalanobis(xm, xp, VI))

# 3. City block metric	
CityBlock_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	CityBlock_d.append(minkowski_distance(xm, xp, 1))

# 5. Chebyshev distance
Chebyshev_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Chebyshev_d.append(distance.chebyshev(xm, xp))

# 6. Cosine distance
Cosine_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Cosine_d.append(distance.cosine(xm, xp))

class Tesla_position:
	def __init__(self, i, x, y, euc, mah, cit, min, che, cos):
		self.i = i
		self.x = x
		self.y = y
		self.euc = euc
		self.mah = mah
		self.cit = cit
		self.min = min
		self.che = che
		self.cos = cos
	def __repr__(self):
		return "%d, %.17f, %.17f, %f, %f, %f, %f, %f, %f\n" % (self.i+1, self.x, self.y, self.euc, self.mah, self.cit, self.min, self.che, self.cos)
	def toString(self):
		return "%d, %.17f, %.17f, %f, %f, %f, %f, %f, %f\n" % (self.i+1, self.x, self.y, self.euc, self.mah, self.cit, self.min, self.che, self.cos)
	def toStringEuc(self):
		return "ID: %d, long: %f, lat: %f, Euclidean distance: %f" % (self.i+1, self.x, self.y, self.euc)
	def toStringMah(self):
		return "ID: %d, long: %f, lat: %f, Mahalanobis distance: %f" % (self.i+1, self.x, self.y, self.mah)
	def toStringCit(self):
		return "ID: %d, long: %f, lat: %f, City block metric: %f" % (self.i+1, self.x, self.y, self.cit)
	def toStringMin(self):
		return "ID: %d, long: %f, lat: %f, Minkowski metric (for p=3): %f" % (self.i+1, self.x, self.y, self.min)
	def toStringChe(self):
		return "ID: %d, long: %f, lat: %f, Chebyshev distance: %f" % (self.i+1, self.x, self.y, self.che)
	def toStringCos(self):
		return "ID: %d, long: %f, lat: %f, Cosine distance: %.12f" % (self.i+1, self.x, self.y, self.cos)
		
pos = []
for i in range(len(xf)):
	pos.append(Tesla_position(i, xf[i], yf[i], Euclidean_d[i], Mahalanobis_d[i], CityBlock_d[i], Minkowski_d[i], Chebyshev_d[i], Cosine_d[i]))
		
print
print 'Sorted by Minkowski metric (for p=3)'
pos.sort(key=attrgetter('min'))
for i in range(10):
	print pos[i].toStringEuc()

nx = []
ny = []
nx_others = []
ny_others = []

for i in range(10):
	nx.append(pos[i].x)
	ny.append(pos[i].y)
	
for i in range(10, len(xf)):
	nx_others.append(pos[i].x)
	ny_others.append(pos[i].y)
	
fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Positions of Tesla")    
ax1.set_xlabel('latitude')
ax1.set_ylabel('longitude')

labels = ['ID={0}'.format(pos[i].i+1) for i in range(10)]

ax1.plot(lat_mean, long_mean, 'bx', label='P')
ax1.plot(nx_others, ny_others, 'bs', label='other points')
ax1.plot(nx, ny, 'ro', label='the 10 closest points\nin Minkowski metric (for p=3)')

legend = ax1.legend(loc='lower right', shadow=True)

vari = 0
anno_length = 20;
for label, x, y in zip(labels, nx, ny):
	if vari%4 == 0:
		plt.annotate(
			label, 
			xy = (x, y), xytext = (anno_length, -anno_length),
			textcoords = 'offset points', ha = 'left', va = 'top',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	elif vari%4 == 1:
		plt.annotate(
			label, 
			xy = (x, y), xytext = (anno_length, anno_length),
			textcoords = 'offset points', ha = 'left', va = 'bottom',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	elif vari%4 == 2:
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-anno_length, -anno_length),
			textcoords = 'offset points', ha = 'right', va = 'top',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	else:	# vari%4 == 3
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-anno_length, anno_length),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	vari += 1
	if vari >= 4:
		vari -= 4
		anno_length += 30
	
		
plt.annotate('Mean', (lat_mean, long_mean), (-20, 20),
	textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()