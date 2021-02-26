import tensorflow as tf


def quaternion_multiply(a, b):
    # c = tf.zeros_like(a)
    # c[:, 0] += a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1] - a[:, 2]*b[:, 2] - a[:, 3]*b[:, 3]
    # c[:, 1] += a[:, 0]*b[:, 1] + a[:, 1]*b[:, 0] + a[:, 2]*b[:, 3] - a[:, 3]*b[:, 2]
    # c[:, 2] += a[:, 0]*b[:, 2] - a[:, 1]*b[:, 3] + a[:, 2]*b[:, 0] + a[:, 3]*b[:, 1]
    # c[:, 3] += a[:, 0]*b[:, 3] + a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1] + a[:, 3]*b[:, 0]

    # c0 = a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1] - a[:, 2]*b[:, 2] - a[:, 3]*b[:, 3]
    # c1 = a[:, 0]*b[:, 1] + a[:, 1]*b[:, 0] + a[:, 2]*b[:, 3] - a[:, 3]*b[:, 2]
    # c2 = a[:, 0]*b[:, 2] - a[:, 1]*b[:, 3] + a[:, 2]*b[:, 0] + a[:, 3]*b[:, 1]
    # c3 = a[:, 0]*b[:, 3] + a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1] + a[:, 3]*b[:, 0]
    # return tf.stack([c0, c1, c2, c3])
    # a, b X * Y * 4
    c0 = a[:, :, 0]*b[:, :, 0] - a[:, :, 1]*b[:, :, 1] - a[:, :, 2]*b[:, :, 2] - a[:, :, 3]*b[:, :, 3]
    c1 = a[:, :, 0]*b[:, :, 1] + a[:, :, 1]*b[:, :, 0] + a[:, :, 2]*b[:, :, 3] - a[:, :, 3]*b[:, :, 2]
    c2 = a[:, :, 0]*b[:, :, 2] - a[:, :, 1]*b[:, :, 3] + a[:, :, 2]*b[:, :, 0] + a[:, :, 3]*b[:, :, 1]
    c3 = a[:, :, 0]*b[:, :, 3] + a[:, :, 1]*b[:, :, 2] - a[:, :, 2]*b[:, :, 1] + a[:, :, 3]*b[:, :, 0]
    return tf.stack([c0, c1, c2, c3], axis=-1)


def quaternion_inverse(a):
    a_star = tf.math.multiply(a, tf.constant([1, -1, -1, -1], dtype=tf.float32))
    a_inv = tf.divide(a_star, tf.math.reduce_sum(tf.square(a), axis=-1, keepdims=True))
    return a_inv


def quaternion_to_axis_normed(a):
    axis = a[:, 1:]
    axis = tf.divide(axis, tf.norm(axis, axis=1, keepdims=True))
    return axis
