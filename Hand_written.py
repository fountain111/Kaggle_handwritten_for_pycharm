import tensorflow as tf
import pandas as pd
import numpy as np
VALIDATION_SIZE = 2000
FLAGS = tf.app.flags.FLAGS
LEARNING_RATE = 1e-4

def split_data():
    datas = pd.read_csv('train.csv')
    labels_flat = datas[[0]].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    images = datas.iloc[:, 1:].values.astype(np.float32)
    labels = one_hot(labels_flat,labels_count)

    # split data into train&corss_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    return train_images,train_labels,validation_images,validation_labels


def inference(datas):
    weight_1 = weight_variable([784, 10])
    bias_1 = bias_variable([10])
    labels = tf.nn.sigmoid(tf.matmul(datas, weight_1) + bias_1)
    return labels

def loss(model_labels,labels):
    return tf.reduce_mean(tf.square(model_labels - labels))

def train(loss):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    return train_step


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def feature_scaling(datas):
    return datas.iloc[:, 1:].values.astype(np.float)


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)