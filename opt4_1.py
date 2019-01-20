# coding: utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
ROW = 32
FEATURE = 2
STEPS = 20000

# init input X and output Y_, y = x1 + x2 +/- 0.05 (bias = 0.05)
rdm = np.random.RandomState(SEED)
X = rdm.rand(ROW, FEATURE)
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 1. declare forward procedure
x = tf.placeholder(tf.float32, shape=(None, FEATURE))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2. declare loss and backward procedure
loss = tf.reduce_mean(tf.square(y_ - y)) # MSE
# train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 3. start session and training
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % ROW
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 500 == 0:
            print("After %d training steps, w1 is:" % i, sess.run(w1))
    print("Final w1 is:", sess.run(w1))

# Using gradient decent optimizer:
# Final w1 is: [[1.0043175 ] [0.99481463]]

# Using momentum optimizer
# Final w1 is: [[1.0043224] [0.994806 ]]

# Using adam optimizer
# Final w1 is: [[1.0044218 ] [0.99464697]]
