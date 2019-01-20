# coding:utf-8

import tensorflow as tf
import numpy as np

# 1. init 2. forward 3. backward 4. iterate training
BATCH_SIZE = 8
SEED = 23455
STEPS = 3000
ROW = 32

rng = np.random.RandomState(SEED)
# init a data set as training set X(32x2) and Y(32x1)
X = rng.rand(ROW, 2)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print(X)
print(Y)

# declare input output and weight
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# forward compute layer
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# backward compute loss
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # print init w1 and w2
    print(sess.run(w1))
    print(sess.run(w2))
    # start training
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % ROW
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print(i, total_loss)
    print(sess.run(w1))
    print(sess.run(w2))

