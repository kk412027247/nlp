# https://www.tensorflow.org/guide/tensor

import tensorflow as tf

v1 = tf.zeros((1, 2))
print(v1)
# tf.Tensor([[0. 0.]], shape=(1, 2), dtype=float32)

v1 = tf.zeros([2, 2])
print(v1)
# tf.Tensor(
# [[0. 0.]
#  [0. 0.]], shape=(2, 2), dtype=float32)

v2 = tf.expand_dims(v1, 1)
print(v2)
# tf.Tensor(
# [[[0. 0.]]
#
#  [[0. 0.]]], shape=(2, 1, 2), dtype=float32)

v3 = tf.zeros((1,))
print(v3)
# tf.Tensor([0.], shape=(1,), dtype=float32)

v4 = tf.zeros((0,))
print(v4)
# tf.Tensor([], shape=(0,), dtype=float32)

v5 = tf.zeros(0)
print(v5)
# tf.Tensor([], shape=(0,), dtype=float32)

v6 = tf.zeros(1)
print(v6)
# tf.Tensor([0.], shape=(1,), dtype=float32)

v7 = tf.zeros(2)
print(v7)
# tf.Tensor([0. 0.], shape=(2,), dtype=float32)

v7 = tf.zeros((2,))
print(v7)
# tf.Tensor([0. 0.], shape=(2,), dtype=float32)

rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# tf.Tensor(4, shape=(), dtype=int32)

rank_1_tensor = tf.constant([1, 2, 3])
print(rank_1_tensor)
# tf.Tensor([1 2 3], shape=(3,), dtype=int32)

rank_2_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(rank_2_tensor)
# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)


t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
print(tf.concat([t1, t2], 0))
# tf.Tensor(
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]], shape=(4, 3), dtype=int32)

print(tf.concat([t1, t2], 1))
# tf.Tensor(
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]], shape=(2, 6), dtype=int32)

t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
print(tf.concat([t1, t2], -1))
# tf.Tensor(
# [[[ 1  2  7  4]
#   [ 2  3  8  4]]
#
#  [[ 4  4  2 10]
#   [ 5  3 15 11]]], shape=(2, 2, 4), dtype=int32)


# If one component of shape is the special value -1,
# the size of that dimension is computed so that the total size remains constant.
# In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
# https://www.tensorflow.org/api_docs/python/tf/reshape?hl=en
t = [[1, 2, 3],
     [4, 5, 6]]
print(tf.reshape(t, [-1]))
# tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32)

print(tf.reshape(t, [3, -1]))
# tf.Tensor(
# [[1 2]
#  [3 4]
#  [5 6]], shape=(3, 2), dtype=int32)

print(tf.reshape(t, [-1, 2]))
# tf.Tensor(
# [[1 2]
#  [3 4]
#  [5 6]], shape=(3, 2), dtype=int32)
