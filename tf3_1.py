import tensorflow as tf

# add two constants, return a tensor
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

# Tensor("add:0", shape=(2,), dtype=float32)
result = a + b
print(result)
