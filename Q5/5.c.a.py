import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kn
from math import*
from decimal import Decimal
from scipy.spatial import distance
from operator import itemgetter, attrgetter, methodcaller

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
		return "%d, %.17f, %.17f, %.17f, %.17f, %.17f, %.17f, %.17f, %.17f\n" % (self.i+1, self.x, self.y, self.euc, self.mah, self.cit, self.min, self.che, self.cos)
		
pos = []
for i in range(len(xf)):
	pos.append(Tesla_position(i, xf[i], yf[i], Euclidean_d[i], Mahalanobis_d[i], CityBlock_d[i], Minkowski_d[i], Chebyshev_d[i], Cosine_d[i]))
	
fileName = 'result.csv'
print '%s is created.' % (fileName)
	
with open(fileName, 'w') as fw:
	fw.write('id, long, lat, euc, mah, cit, min, che, cos\n')
	for i in range(len(xf)):
		fw.write(pos[i].toString())
