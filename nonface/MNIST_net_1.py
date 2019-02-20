import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
layersize = 100
lr=.01
ep=20
n_h1 = 200
n_h2 = 100

n_output= 10
batch_size = 100

x = tf.placeholder('float', [None, 784], name='data')
y = tf.placeholder('float', name='answers')



def neural_network_model(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([784, n_h1]),name='w1'),
                      'biases':tf.Variable(tf.random_normal([n_h1]),name='b1')}
    #w1=tf.summary.histogram('w1',hl_1['weights'])

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_h1, n_h2]),name='w2'),
                      'biases':tf.Variable(tf.random_normal([n_h2]),name='b2')}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_h2, n_output]),name='w3'),
                    'biases':tf.Variable(tf.random_normal([n_output]),name='b3')}

    l1 = tf.add(tf.matmul(data,hl_1['weights'],name='dataXw1'), hl_1['biases'],name='z1')
    l1 = tf.nn.sigmoid(l1,name='a1')

    l2 = tf.add(tf.matmul(l1,hl_2['weights'],name='a1Xw2'), hl_2['biases'],name='z2')
    l2 = tf.nn.sigmoid(l2,name='a2')

    output = tf.matmul(l2,output_layer['weights'],name='a2Xw3') + output_layer['biases']

    return output

merged = tf.summary.merge_all()

def train_neural_network(x):
    prediction = neural_network_model(x)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) #finds cost
    with tf.name_scope('Optimize'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost) #an optimizer that minimizes loss
    hm_epochs = ep
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('C:/Code/mnist/dense1', sess.graph) #tensorboard --logdir C:/Code/mnist
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _,c= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            #print('Epoch', epoch, 'completed out of',hm_epochs,'cost:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc=accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
       print('Accuracy:',acc)
        # summ , _ = sess.run([merged, cost], feed_dict={x: epoch_x, y: epoch_y})
        # writer.add_summary(summ)
        writer.close()
        return {'acc':acc, 'pred':prediction}

def inverse

net=train_neural_network(x)
