from numpy import *


votes = zeros((2000 , 10))
labels = zeros(2000)

for i in range(0,10):
	votes[:,i]  = loadtxt('./rslts/'+str(i+1)+'_test.txt')
	

#votes[votes > 0.5] = 1

votes = sum(votes,1)/10

labels[votes <  0.5] = 1
labels[votes >=  0.5] = 2

print('total = ' , 2000)
print('pos = ' , sum(labels==2))
print('neg = ' , sum(labels==1))

savetxt('labels.csv' , labels , fmt='%i')
