# https://www.tensorflow.org/guide/tensor

import tensorflow as tf

v1 = tf.zeros((1, 2))
print(v1)

v1 = tf.zeros([1, 2])
print(v1)

v2 = tf.expand_dims(v1, 1)
print(v2)

v3 = tf.zeros((1,))
print(v3)

v4 = tf.zeros((0,))
print(v4)

v5 = tf.zeros(0)
print(v5)

v6 = tf.zeros(1)
print(v6)

v7 = tf.zeros(2)
print(v7)

v7 = tf.zeros((2,))
print(v7)

rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([1, 2, 3])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(rank_2_tensor)

