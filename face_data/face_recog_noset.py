import tensorflow as tf
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from random import shuffle

addrs = glob.glob('C:\Code\TensorFlow practice\\face_data\*.jpg')
#for i in glob.glob('C:\Code\TensorFlow practice\\face_data\*.jpg'):
#    addrs.append(i)
labels = []
for i in addrs:
    if 'ben' in i:
        labels.append([1,0,0])
    elif 'james' in i:
        labels.append([0,1,0])
    elif 'zak' in i:
        labels.append([0,0,1])

c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

train_addrs = addrs[0:int(.7*len(addrs))]
train_labels = labels[0:int(.7*len(labels))]

test_addrs = addrs[int(0.7*len(addrs)):]
test_labels = labels[int(0.7*len(labels)):]


def loadImg(adrs):
    img = cv2.imread(adrs,0)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float64)
    return img
trainData=[]
testData=[]
for i in range(len(train_addrs)):
    trainData.append(loadImg(train_addrs[i]).reshape(100*100))
for i in range(len(test_addrs)):
    testData.append(loadImg(test_addrs[i]).reshape(100*100))


#=============================================================================================================
#with tf.device("/gpu:0"):
tf.logging.set_verbosity(0)
print('====================================================================================================')
print('Ben Shakow:')
epochs=20
batch_size=100
act=tf.nn.sigmoid
L1_size=1000
L2_size=500
C1_filters=64
C2_filters=100
input = tf.placeholder('float64',[None,100*100],name='Data')
trueVal = tf.placeholder('float64', name='trueval')

def makenet(input):
    inputs = tf.reshape(input, [-1, 100, 100, 1])
    C1 = tf.layers.conv2d(inputs=inputs,filters=C1_filters,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    P1 = tf.layers.max_pooling2d(inputs=C1, pool_size=[2, 2], strides=2)
    C2 = tf.layers.conv2d(inputs=P1,filters=C2_filters,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    P2 = tf.layers.max_pooling2d(inputs=C2, pool_size=[2, 2], strides=2)
    flat = tf.reshape(P2, [-1, 25 * 25 * C2_filters])
    L1 = tf.layers.Dense(units=L1_size,activation=act,name='L1')
    L2 = tf.layers.Dense(units=L2_size,activation=act,name='L2')
    dout = tf.layers.dropout(inputs=L2(L1(flat)), rate=0.4)
    classes = tf.layers.Dense(units=3,activation=act,name='classes')
    predVal=classes(dout)
#    predVal=classes()
    return predVal


def net(input,read):
    with tf.device("/gpu:0"):
        predVal=makenet(input)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=predVal, labels=trueVal) )
        with tf.name_scope('Optimize'):
            optimizer = tf.train.GradientDescentOptimizer(.01)
            train = optimizer.minimize(loss)
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement = True, gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        # if(read):
        #     saver.restore(sess, "C:/Code/faces/saved")
        #     print("Restored")
        # else:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('C:/Code/faces/graph', sess.graph)
        #for epoch in range(epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(1):
                epoch_x = trainData
                epoch_y = train_labels
                _,c= sess.run([train, loss], feed_dict={input: epoch_x, trueVal: epoch_y})
                epoch_loss += c
            print('Epoch:', epoch,',','cost:',100*epoch_loss)
        correct = tf.equal(tf.argmax(predVal, 1), tf.argmax(trueVal, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc=accuracy.eval({input:testData, trueVal:test_labels})
        print('Accuracy:',acc)
        #    saver.save(sess,"C:/Code/faces/saved")
        #    print("Saved")
        writer.close()
        sess.close()
        return(acc)
avacc=0
numruns=1
for i in range(numruns):
    avacc+=net(input,False)
avacc = avacc/numruns
print(avacc)
print(avacc)
print(avacc)
print(avacc)
print(avacc)
