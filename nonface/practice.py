import os
#import keyboard
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
with tf.device("/gpu:0"):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

    tf.logging.set_verbosity(tf.logging.ERROR)
    print('====================================================================================================')
    print('Ben Shakow:')
    epochs=20
    batch_size=100
    act=tf.nn.relu
    L1_size=200
    L2_size=100
    C1_filters=15
    C2_filters=25
    input = tf.placeholder('float64',[None,784],name='Data')
    trueVal = tf.placeholder('float64', name='trueval')

def makenet(input):
    inputs = tf.reshape(input, [-1, 28, 28, 1])
    C1 = tf.layers.conv2d(inputs=inputs,filters=C1_filters,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    P1 = tf.layers.max_pooling2d(inputs=C1, pool_size=[2, 2], strides=2)
    C2 = tf.layers.conv2d(inputs=P1,filters=C2_filters,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    P2 = tf.layers.max_pooling2d(inputs=C2, pool_size=[2, 2], strides=2)
    flat = tf.reshape(P2, [-1, 7 * 7 * C2_filters])
    L1 = tf.layers.Dense(units=L1_size,activation=act,name='L1')
    L2 = tf.layers.Dense(units=L2_size,activation=act,name='L2')
    dout = tf.layers.dropout(inputs=L2(L1(flat)), rate=0.4)
    classes = tf.layers.Dense(units=10,activation=act,name='classes')
    predVal=classes(dout)
#    predVal=classes();
    return predVal
def net(input,read,maxacc):
    with tf.device("/gpu:0"):
        predVal=makenet(input)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=predVal, labels=trueVal) )
        with tf.name_scope('Optimize'):
            optimizer = tf.train.GradientDescentOptimizer(.01)
            train = optimizer.minimize(loss)
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement = True, gpu_options=tf.GPUOptions(allow_growth=True))
#    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if(read):
            saver.restore(sess, "C:/Code/mnist/homemade_mnist/Saved")
            print("Restored")
        else:
            sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('C:/Code/mnist/homemade_mnist', sess.graph)
        #for epoch in range(epochs):
        epoch_loss = 1000
        epoch=0
        while(epoch_loss>30):
            if(epoch>5):
                break
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _,c= sess.run([train, loss], feed_dict={input: epoch_x, trueVal: epoch_y})
                epoch_loss += c
            print('Epoch:', epoch,',','cost:',epoch_loss)
            epoch+=1;
        correct = tf.equal(tf.argmax(predVal, 1), tf.argmax(trueVal, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc=accuracy.eval({input:mnist.test.images, trueVal:mnist.test.labels})
        print('Accuracy:',acc)
        if(acc>maxacc):
            saver.save(sess,"C:/Code/mnist/homemade_mnist/Saved")
            print("Saved")
        writer.close()
        return(acc)
maxacc=0.0
while(maxacc<1):
    acc=net(input,True,maxacc)
    if(maxacc<acc):
        maxacc=acc
