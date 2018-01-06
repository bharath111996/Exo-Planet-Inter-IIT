from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from numpy import *
import pickle
import os
import matplotlib.pyplot as plt
import scipy.signal as filt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#read csv data
import csv

numTest = 2000

#read training data
fi = csv.reader(open('train.csv'))
realData = zeros((60,3197))
falseData = zeros((3500,3197))
count = 0 
for row in fi:
    if count > 0  and count <= 30:
        row = array(row)
        row = row.astype(float)
        realData[count-1] = row[1:]
    if count > 30  and count < 3530:
        row = array(row)
        row = row.astype(float)
        if(count <60):
            realData[count] = row[1:]
            #print(mean(realData[count]))
        falseData[count-30] = filt.medfilt(row[1:],31)
        
    if count > 3530:
        break 
    count = count + 1

#read test data
realDataTest = zeros((2000,3197))
count  = 0
ft = csv.reader(open('test.csv'))
for row in ft:
    if count > 0  and count <= 2000:
        row = array(row[1:])
        row = row.astype(float)
        realDataTest[count-1] = row
        
    if count > 3530:
        break 
    count = count + 1
    

FLAGS = None
with open('./pickle_data/transit_data_train.pkl', 'rb') as f:
    data = pickle.load(f)        

data = data['results']
num = 3500      

dataPos = data[:,1]
dataNeg = data[:,2]


def weight_init(shape):
    x = tf.truncated_normal(shape , mean = 0.0, stddev = sqrt(2.0/shape[0]))
    return tf.Variable(x)
 
def bias_init(shape):
    x = tf.constant(0.01,shape = shape)
    return tf.Variable(x)

def next_batch(s):
    x = zeros((s,180))
    y = zeros((s,1))
    r = random.permutation(num-500)
    for i in range(0,int(s/2)):
        x[i,:] = dataPos[r[i]]
        x[i,:] = (x[i,:] - mean(x[i,:]))/(0.001+std(x[i,:]))
        y[i] = 1

    r = random.permutation(num-500)
    r[r<30] = 31    
    for i in range(int(s/2),s):
        t1 = falseData[r[i]]
        tt = t1[0:179*17]
        tt = reshape(tt,(179,17))
        tt = mean(tt , 1)
        ttt = zeros(180)
        ttt[0:179] = tt        
        x[i,:] = ttt
        x[i,:] = (x[i,:]-mean(x[i,:]))/(0.001+std(x[i,:]))
        #plt.plot(x[i])
        #plt.show()    
        
        y[i] = 0   
    return x,y

def valid_batch(s):
    x = zeros((s,180))
    y = zeros((s,1))
    r = random.permutation(500)+3000
    for i in range(0,int(s/2)):
        x[i,:] = dataPos[r[i]]
        x[i,:] = (x[i,:] - mean(x[i,:]))/(0.001+std(x[i,:]))
        y[i] = 1

    r = random.permutation(500)+3000    
    for i in range(int(s/2),s):
        t1 = falseData[r[i]]
        tt = t1[0:179*17]
        tt = reshape(tt,(179,17))
        tt = mean(tt , 1)
        ttt = zeros(180)
        ttt[0:179] = tt        
        x[i,:] = ttt
        x[i,:] = (x[i,:]-mean(x[i,:]))/(0.001+std(x[i,:]))

        y[i] = 0   
    return x,y

def binData(arr,n):
    binW = int(len(arr)/n)
    arr = arr[0:binW*n]
    temp = zeros(binW)    
    for i in range(0,binW*n):
        temp[i%binW] = temp[i%binW] + arr[i]

    return temp

def main(_):
  # Create the model
  x = tf.placeholder(tf.float32, [None, 180])
  w1 = weight_init([180, 64])
  b1 = bias_init([64])
  ll1 = tf.matmul(x, w1) + b1
  l1 = tf.nn.tanh(ll1)
  
  w2 = weight_init([64, 32])
  b2 = bias_init([32])
  ll2 = tf.matmul(l1, w2) + b2
  l2 = tf.nn.tanh(ll2)

  w3 = weight_init([32, 8])
  b3 = bias_init([8])
  ll3 = tf.matmul(l2, w3) + b3
  l3 = tf.nn.tanh(ll3)

  w4 = weight_init([8,1])
  b4 = bias_init([1])
  y = tf.matmul(l3, w4) + b4
      
	
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  cross_entropy = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  # Test trained model
  correct_prediction = tf.equal(tf.greater(y,0), tf.greater(y_,0.5))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  numIter = 2000
  ens = 10
  err = zeros((numIter))
  # Train
  for i in range(numIter):
    batch_xs, batch_ys = next_batch(100)
    batch_vs, batch_ws = valid_batch(100)
    
    err = sess.run(accuracy, feed_dict={x: batch_vs,
                                      y_: batch_ws})
   # print(err[0] ,' and ', err[99])
    if(i == numIter-1):
        probab = zeros(60)
        for j in range(0,60):
            temp = realData[j]
            tt = temp[0:179*17]
            tt = reshape(tt,(179,17))
            tt = mean(tt , 1)
            ttt = zeros(180)
            ttt[0:179] = tt        
            
            loc = ttt - mean(ttt)
            loc = loc/(std(loc)+0.001)
            loc = reshape(loc , (1,180))
            tt = sess.run(y, feed_dict={x: loc , y_:ones((1,1))}) 	   
            probab[j] = 1./(1+exp(-tt))
        savetxt('./rslts/'+str(ens)+'_train.txt' , probab)
        
        probab = zeros(numTest)
        
        for j in range(0,numTest):
            temp = realDataTest[j]
            tt = temp[0:179*17]
            tt = reshape(tt,(179,17))
            tt = mean(tt , 1)
            ttt = zeros(180)
            ttt[0:179] = tt        
            
            loc = ttt - mean(ttt)
            loc = loc/(std(loc)+0.001)
            loc = reshape(loc , (1,180))
            tt = sess.run(y, feed_dict={x: loc , y_:ones((1,1))}) 	   
            probab[j] = 1./(1+exp(-tt))
        savetxt('./rslts/'+str(ens)+'_test.txt' , probab)                
        
    if(i % 25 == 0):
        print('iteration = ',i,' accuracy = ',err*100)       
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
