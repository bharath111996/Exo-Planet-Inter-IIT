from numpy import *
import pickle
import os
import matplotlib.pyplot as plt
import scipy.signal as filt

#read csv data
import csv

def gval(x,m,s):
	val = x-m
	return (1/sqrt(2*pi))*(1/s)*exp(-(val**2)/(2*s**2))


fi = csv.reader(open('train.csv'))
realData = zeros((3500,3197))
r = zeros(3500)
count = 0 
for row in fi:
    if count > 0  and count <= 3500:
        row = array(row)
        row = row.astype(float)
        realData[count-1] = filt.medfilt(row[1:],31)
        t = abs(realData[count-1])
        med =  median(t)
        medL = median(t[t<med])
        medR = median(t[t>med])
        r[count-1] = medR - medL
       # print('count= ',count, ' med = ',medR - medL)
    if count > 3500:
        break 
    count = count + 1

mu1 = mean(r[0:30])
sig1 = std(r[0:30])

mu2 = mean(r[30:])
sig2 = std(r[30:])


print(mu1)
print(sig1)
print(mu2)
print(sig2)

ft = csv.reader(open('test.csv'))
realData = zeros((2000,3197))
p1 = zeros(2000)
p2 = zeros(2000)

count = 0 
for row in ft:
    if count > 0  and count <= 2000:
    	row = array(row[1:])
        row = row.astype(float)
        realData[count-1] = filt.medfilt(row,31)
        t = abs(realData[count-1])
        med =  median(t)
        medL = median(t[t<med])
        medR = median(t[t>med])
        p1[count-1] = gval(medR-medL , mu1,sig1)
        p2[count-1] = gval(medR-medL , mu2,sig2)
 	
    if count > 2000:
        break 
    count = count + 1

savetxt('p1.txt' , p1)
savetxt('p2.txt' , p2)
	

plt.plot(r[0:30],'ro')
plt.plot(r[30:],'bo')
plt.show()

