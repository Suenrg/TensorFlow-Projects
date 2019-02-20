#Ben Shakow

import tensorflow as tf
import numpy as np
import glob
import cv2

addrs = glob.glob('*.jpg')
labels = [0 if 'ben' in adrs else 1 for adrs in addrs]

train_addrs = addrs[0:int(1*len(addrs))]
train_labels = labels[0:int(1*len(labels))]

# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(labels)):]

def loadImg(adrs):
    img = cv2.imread(adrs,0)
#    img = cv2.Canny(img,100,200)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

trecordName = 'dataset.tfrecords'
writer = tf.python_io.TFRecordWriter(trecordName)
m=0
for i in train_addrs:
    img = loadImg(i)
    label=train_labels[m]
    m+=1
    feature = {'train/label': _int64_feature(label),'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

# terecordName = 'ftest.tfrecords'
# writer = tf.python_io.TFRecordWriter(terecordName)
# m=0
# for i in test_addrs:
#     img = loadImg(i)
#     label=test_labels[m]
#     m+=1
#     feature = {'test/label': _int64_feature(label),'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(example.SerializeToString())
# writer.close()
