import tensorflow as tf
import numpy as np


def gauss_kl_fixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)


def gauss_log_prob(mu, logstd, x):
    # probability to take action x, given paramaterized guassian distribution
    var = tf.exp(2 * logstd)
    gp = - tf.square(x - mu) / (2 * var) \
        - .5 * tf.log(tf.constant(2 * np.pi)) \
        - logstd
    return tf.reduce_sum(gp, [1])


def gauss_KL(mu1, logstd1, mu2, logstd2, axis=None):
    # KL divergence between two paramaterized guassian distributions
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(
        logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5,
        axis=axis
    )
    return kl


def gauss_ent(mu, logstd):
    # shannon entropy for a paramaterized guassian distributions
    h = tf.reduce_sum(
        logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32)
    )
    return h


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(
        [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0
    )


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        shapes = map(var_shape, var_list)
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)


def flatten(tensors):
    if isinstance(tensors, (tuple, list)):
        return tf.concat(
            tuple(tf.reshape(tensor, [-1]) for tensor in tensors), axis=0)
    else:
        return tf.reshape(tensors, [-1])


class unflatten(object):
    def __init__(self, tensors_template):
        self.tensors_template = tensors_template

    def __call__(self, colvec):
        if isinstance(self.tensors_template, (tuple, list)):
            offset = 0
            tensors = []
            for tensor_template in self.tensors_template:
                sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
                tensor = tf.reshape(
                    colvec[offset:(offset + sz)],
                    tensor_template.shape
                )
                tensors.append(tensor)
                offset += sz

            tensors = list(tensors)
        else:
            tensors = tf.reshape(colvec, self.tensors_template.shape)

        return tensors
