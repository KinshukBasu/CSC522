import numpy as np
from math import*
from decimal import Decimal
from scipy.spatial import distance
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pyplot as plt


def MahaDist(xdata,ydata,xmean,ymean):

	res = []

	for i in range(0,len(xdata)):
		x[i] = xdata[i]-ymean
		y[i] = ydata[i]-ymean



	S = np.matrix(np.cov(x,y, rowvar=False))

	for i in range(0, len(x)):
		
		p = np.matrix([x[i],y[i]])
		res.append(float(np.sqrt(p*S*(np.matrix.transpose(p)))) )

	return(res)


def CityBlock(xf,yf,xmean,ymean):

	res = []
	for i in range(0,len(xf)):

		s = abs(xf[i]-xmean) + abs(yf[i]-ymean)
		res.append(s)
	return(res)


def Minkowski(xf,yf,xmean,ymean):

	res = []
	for i in range(0, len(xf)):
		s = (abs(xf[i]-xmean)**3 + abs(yf[i]-ymean)**3)** (1. / 3)
		res.append(s)
	return res


def Chebyshev(xf,yf,xmean,ymean):
	res = []
	for i in range(0, len(xf)):
		s = max(abs(xf[i]-xmean), abs(yf[i]-ymean))
		res.append(s)
	return(res)

def Cosine(xf,yf,xmean,ymean):
	res = []
	for i in range(0, len(xf)):
		s = ((xf[i]*xmean)+(yf[i]*ymean))/( (np.sqrt(xf[i]**2 + yf[i]**2))*(np.sqrt(xmean**2 + ymean**2)) )
		res.append(s)
		
	return(res)
	#This function can return negative values as well, Wikipedia says so. If positive is required, use abs() for numerator









x = []
y=[]
with open("locations.csv") as f: 
	first_line = f.readline()	# skip first line
	for line in f:
		x.append(line.split(',')[1])
		y.append(line.split(',')[2])

xf = [float(xi) for xi in x]
yf = [float(yi) for yi in y]

xmean = sum(xf)/len(xf)
ymean = sum(yf)/len(yf)

mahadist = MahaDist(xf,yf,xmean,ymean)
print len(mahadist)

cityblock = CityBlock(xf,yf,xmean,ymean)
print len(cityblock)

minkowski = Minkowski(xf,yf,xmean,ymean)
print len(minkowski)

chebyshev = Chebyshev(xf,yf,xmean,ymean)
print len(chebyshev)

cosine = Cosine(xf,yf,xmean,ymean)
print len(cosine)