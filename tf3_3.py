# coding:utf-8

import tensorflow as tf

# input and weight, forward, get result

# constant
x = tf.constant([[0.7, 0.5]])
# random variable
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# random variable
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# a = w1 * x
a = tf.matmul(x, w1)
# y = w2 *a
y = tf.matmul(a, w2)

print(x)
print(w1)
print(w2)
print(a)
print(y)

with tf.Session() as sess:
    # init all variables
    init_op = tf.global_variables_initializer()
    # run session
    sess.run(init_op)
    print("y in tf3_3.py is:\n", sess.run(y))
