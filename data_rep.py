


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def Normalize(y,mean,stdev):
    return (y-mean)/(stdev+1e-5)
def DeNormalize(y,mean,stdev):
    return y*(stdev+1e-5)+mean
def Xavier(n1,n2):
    no = np.sqrt(4.0/(n1+n2))
    return no
def NewVariable(size,marker,datatype = tf.float32):
    xav = Xavier(size[0],size[1])
    if marker=='weight':
        return tf.Variable(tf.random_uniform(size,-xav,xav),dtype = datatype)
    elif marker=='bias':
        return tf.Variable(tf.random_uniform([size[1]],-xav,xav),dtype = datatype)



x = tf.placeholder(tf.float32,shape = [None,3197])
y_ = tf.placeholder(tf.float32,shape = [None,2])
pkeep = tf.placeholder(tf.float32)
pkeep_1 = 1.0

# define K L M N according to our need
K = 2500
L = 100
M = 75
N = 30

w1 = tf.Variable(tf.truncated_normal([3197,K],stddev=0.1))
b1 = tf.Variable(tf.ones([K])/10)

w2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
b2 = tf.Variable(tf.ones([L])/10)

w3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
b3 = tf.Variable(tf.ones([M])/10)

w4 = tf.Variable(tf.truncated_normal([M,0],stddev=0.1))
b4 = tf.Variable(tf.ones([0]))

w5 = tf.Variable(tf.truncated_normal([0,2],stddev=0.1))
b5 = tf.Variable(tf.zeros([2]))

init = tf.global_variables_initializer()

Y1 = tf.nn.relu(tf.matmul(x,w1) + b1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d,w2) + b2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(Y2d,w3) + b3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.relu(tf.matmul(Y3d,w4) + b4)

Y  = tf.nn.softmax(tf.matmul(Y4,w5) + b5)


#loss function
cross_entropy = -tf.reduce_sum(y_ * tf.log(Y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(1-y_)*tf.log(1-Y))


# percentage of correct ans in a batch
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
'''

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=Y))
cross_entropy = cost+0.0000001*tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()])
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
recall = tf.metrics.recall(y_,Y)
'''
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy)


A=np.genfromtxt('exoTrain.csv',delimiter = ',')
xtrain = A[1:,1:]
ytrain = A[1:,0]
A=np.genfromtxt('exoTest.csv',delimiter = ',')
xtest = A[1:,1:]
ytest = A[1:,0]
xdiv = np.max(xtrain)-np.min(xtrain)
xmean = np.min(xtrain)
xtrain = Normalize(xtrain,xmean,xdiv)
xtest = Normalize(xtest,xmean,xdiv)
ytrain = ytrain-1
ytest = ytest - 1

ytest = np.reshape(ytest,[-1,1])
ytest = np.hstack((ytest,(ytest+1)/2))
ytrain = np.reshape(ytrain,[-1,1])
ytrain = np.hstack((ytrain,(ytrain+1)/2))

print(xtrain.shape,ytrain.shape)

# for r in [1500]:
#     for i in range(37):
#         temp = np.array([np.concatenate([xtrain[i,1000:],xtrain[i,:1000]])])
#         xtrain = np.concatenate((xtrain,temp),axis=0)

# for i in range(37):
#     temp = np.array([np.flip(xtrain[i,:],0)])
#     xtrain = np.concatenate((xtrain,temp),axis=0)
#
# dim = 37*1
#
# ytrain = np.concatenate((ytrain,np.concatenate(([np.ones(dim)],[np.ones(dim)])).T))

print(xtrain.shape,ytrain.shape)

sess = tf.Session()
sess.run(init)

for _ in range(10):
    #loading of batch
    train_data = {x : np.reshape(xtrain,[-1,3197]), y_:np.reshape(ytrain,[-1,2]) ,pkeep:pkeep_1}
    sess.run(train_step ,feed_dict=train_data)
    a = sess.run(accuracy ,feed_dict=train_data)
    c = sess.run(cross_entropy ,feed_dict=train_data)

    print("train accuracy = " ,a ,"train cost = " ,c)
    #what is use of np.reshape

    test_data = {x:np.reshape(xtest,[-1,3197]) , y_:np.reshape(ytest,[-1,2]) ,pkeep:1}
    a = sess.run(accuracy ,feed_dict=test_data)
    c = sess.run(cross_entropy ,feed_dict=test_data)
    print ('test accuracy =' , a ,"test cost =", c)

Test_predict = sess.run(Y,test_data)
print (Test_predict.shape)
plt.figure(1)
r=np.arange(Test_predict.shape[0])
plt.plot(r,Test_predict[:,0],'r')
plt.plot(r,ytest[:,0],'g')
plt.figure(2)
r=np.arange(Test_predict.shape[0])
plt.plot(r,Test_predict[:,1],'r')

plt.plot(r,ytest[:,1],'g')
plt.show()
