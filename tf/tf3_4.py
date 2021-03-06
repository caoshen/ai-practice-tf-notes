# coding:utf-8

import tensorflow as tf

# input and weight using placeholder
x = tf.placeholder(tf.float32, [1, 2])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # feed dict to x and get y
    print("y in tf3_4.py is:\n", sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
