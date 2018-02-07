"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os
import numpy as np

import tensorflow as tf
#import tensorflow.examples.tutorials.mnist.input_data as input_data
from attacks import fgm, deepfool, jsma
from networks import residual_network as model
from utils import *

IMG_SIZE = 32
IMG_CHAN = 3
n_classes = 10

#mnist = input_data.read_data_sets('/home/chnlkw/data/MNIST_data/', one_hot=True)
trainx, trainy, testx, testy = load_all('/home/chnlkw/data/cifar-10-batches-py/')

"""
def model(x, logits=False, training=False, reuse=True, name='model'):
    with tf.variable_scope(name) as scope:
        if reuse==True:
            scope.reuse_variables()
        with tf.variable_scope('conv0'):
            z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
            z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

        with tf.variable_scope('conv1'):
            z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
            z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

        with tf.variable_scope('flatten'):
            z = tf.layers.flatten(z)

        with tf.variable_scope('mlp'):
            z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
            z = tf.layers.dropout(z, rate=0.25, training=training)

        logits_ = tf.layers.dense(z, units=10, name='logits')
        y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y
"""
class env(object):
    def __init__(self, sess, batch_size=128):
        self.sess = sess
        self.batch_size = batch_size
        self.build_model()
    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, IMG_CHAN], name='x')
        self.y = tf.placeholder(tf.float32, [None, n_classes], name='y')
        self.adv_y = tf.placeholder(tf.int32, (), name='adv_y')
        self.training = tf.placeholder_with_default(False, (), name='mode')
        self.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
        self.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')

        self.ybar, logits = model(self.x, logits=True, training=self.training, reuse=False)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.ybar, axis=1))
            self.acc = tf.reduce_sum(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            self.loss = tf.reduce_sum(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        self.x_fgsm = fgm(model, self.x, epochs=self.adv_epochs, eps=self.adv_eps)
        self.x_deepfool = deepfool(model, self.x, epochs=self.adv_epochs, batch=True)
        self.x_jsma = jsma(model, self.x, self.adv_y, eps=self.adv_eps, epochs=self.adv_epochs)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self, X_data, y_data, X_valid=None, y_valid=None, load=False, epochs=50, name='model'):
        if load:
            self.saver.restore(self.sess, 'model/{}'.format(name))
        n = X_data.shape[0]
        n_batch = int((n+self.batch_size-1) / self.batch_size)

        for epoch in range(epochs):
            print('\nEpoch {0}/{1}'.format(epoch+1, epochs))

            ind = np.arange(n)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

            for batch in range(n_batch):
                start = batch * self.batch_size
                end = min(n, start + self.batch_size)
                _, loss, acc= self.sess.run([self.train_op, self.loss, self.acc], feed_dict={self.x: X_data[start:end], self.y:y_data[start:end], self.training:True})

                print('batch {0}/{1}/{2:.4f}/{3:.4f}'.format(batch+1, n_batch, loss, acc), end='\r')
            if X_valid is not None:
                self.evaluate(X_valid, y_valid)
            print('\nSaving model')
            os.makedirs('model', exist_ok=True)
            self.saver.save(self.sess, 'model/{}'.format(name))
    def evaluate(self, X_data, y_data):
        n = X_data.shape[0]
        n_batch = int((n+self.batch_size-1) / self.batch_size)
        loss = 0
        acc = 0
        for batch in range(n_batch):
            print('batch {0}/{1}'.format(batch+1, n_batch), end='\r')
            start = batch * self.batch_size
            end = min(n, start + self.batch_size)
            batch_loss, batch_acc = self.sess.run([self.loss, self.acc], feed_dict={self.x: X_data[start:end], self.y:y_data[start:end]})
            loss += batch_loss
            acc += batch_acc
        loss /= n
        acc /= n

        print('loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
        return loss, acc

    def predict(self, X_data):
        n = X_data.shape[0]
        n_batch = int((n+self.batch_size-1) / self.batch_size)
        yval = np.empty((n, n_classes))

        for batch in range(n_batch):
            start = batch * self.batch_size
            end = min(n, start + self.batch_size)
            y_batch = self.sess.run(self.ybar, feed_dict={self.x: X_data[start:end]})
            yval[start:end] = y_batch
        print()
        return yval

    def make_fgsm(self, X_data, epochs=1, eps=0.01):
        n = X_data.shape[0]
        n_batch = int((n + self.batch_size - 1) / self.batch_size)
        X_adv = np.empty_like(X_data)

        for batch in range(n_batch):
            print(batch)
            start = batch * self.batch_size
            end = min(n, start + self.batch_size)
            adv = self.sess.run(self.x_fgsm, feed_dict={self.x: X_data[start:end], self.adv_eps: eps, self.adv_epochs: epochs})
            X_adv[start:end] = adv
        print()
        return X_adv

    def make_deepfool(self, X_data, epochs=1, eps=0.01):
        n = X_data.shape[0]
        n_batch = int((n + self.batch_size - 1) / self.batch_size)
        X_adv = np.empty_like(X_data)

        for batch in range(n_batch):
            print(batch)
            start = batch * self.batch_size
            end = min(n, start + self.batch_size)
            adv = self.sess.run(self.x_deepfool, feed_dict={self.x: X_data[start:end], self.adv_epochs: epochs})
            X_adv[start:end] = adv
        print()
        return X_adv

    def make_jsma(self, X_data, epochs=0.2, eps=1.0):
        n = X_data.shape[0]
        n_batch = int((n + self.batch_size - 1) / self.batch_size)
        X_adv = np.empty_like(X_data)

        for batch in range(n_batch):
            print(batch)
            start = batch * self.batch_size
            end = min(n, start + self.batch_size)
            adv = self.sess.run(self.x_jsma, feed_dict={self.x: X_data[start:end], self.adv_eps: eps, self.adv_epochs: epochs, self.adv_y: np.random.choice(n_classes)})
            X_adv[start:end] = adv
        print()
        return X_adv
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
model = env(sess)
#X_train = mnist.train.images.reshape([-1, IMG_SIZE,  IMG_SIZE, IMG_CHAN])
#y_train = mnist.train.labels
#X_test = mnist.test.images.reshape([-1, IMG_SIZE,  IMG_SIZE, IMG_CHAN])
#y_test = mnist.test.labels
#X_validation = mnist.validation.images.reshape([-1, IMG_SIZE,  IMG_SIZE, IMG_CHAN])
#y_validation = mnist.validation.labels

model.train(trainx, trainy)
adv_fgsm = model.make_fgsm(testx, epochs=12, eps=0.01)
adv_deepfool = model.make_deepfool(testx)
adv_jsma = model.make_jsma(testx)
model.evaluate(adv_fgsm, testy)
model.evaluate(adv_deepfool, testy)
model.evaluate(adv_jsma, testy)
