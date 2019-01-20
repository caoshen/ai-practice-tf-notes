import tensorflow as tf

# matrix multiply w and x
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x, w)

# Tensor("MatMul:0", shape=(1, 1), dtype=float32)
print(y)

# [[11.]]
with tf.Session() as sess:
    print(sess.run(y))
