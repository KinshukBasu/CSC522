import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kn
from math import*
from decimal import Decimal
from scipy.spatial import distance
from operator import itemgetter, attrgetter, methodcaller

x =[]
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

Euclidean_d = []

for i in xrange(len(xf)):
	Euclidean_d.append(np.sqrt((xf[i]-lat_mean)**2 + (yf[i]-long_mean)**2))

#print Euclidean_d[:3]

# Mahalanobis distance

# Ackowledge: used from http://kldavenport.com/mahalanobis-distance-and-outliers/
def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])
    
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md

Mahalanobis_d = MahalanobisDist(xf, yf)
#print Mahalanobis_d[:3]

# City block metric

# Ackowledge: used from http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
def manhattan_distance(x,y):
	return sum(abs(a-b) for a,b in zip(x,y))
	
CityBlock_d = []

xp = [lat_mean, long_mean]
for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	CityBlock_d.append(manhattan_distance(xm, xp))
#print CityBlock_d[:3]

# Minkowski metric (for p=3)

# Ackowledge: used from http://dataconomy.com/implementing-the-five-most-popular-similarity-measures-in-python/
def nth_root(value, n_root):
	root_value = 1/float(n_root)
	return round (Decimal(value) ** Decimal(root_value),3)

def minkowski_distance(x, y, p_value):
	return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
	
Minkowski_d = []

for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Minkowski_d.append(minkowski_distance(xm, xp, 3))
#print Minkowski_d[:30]

# Chebyshev distance

Chebyshev_d = []
for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Chebyshev_d.append(distance.chebyshev(xm, xp))
#print Chebyshev_d[:30]

# Cosine distance
Cosine_d = []
for i in range(len(xf)):
	xm = [xf[i], yf[i]]
	Cosine_d.append(distance.cosine(xm, xp))
#print Cosine_d[:10]

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
		
pos = []
for i in range(len(xf)):
	pos.append(Tesla_position(i, xf[i], yf[i], Euclidean_d[i], Mahalanobis_d[i], CityBlock_d[i], Minkowski_d[i], Chebyshev_d[i], Cosine_d[i]))
	
fileName = 'result.csv'
print '%s is created.' % (fileName)
	
with open(fileName, 'w') as fw:
	fw.write('id, long, lat, euc, mah, cit, min, che, cos\n')
	for i in range(len(xf)):
		fw.write(pos[i].toString())
