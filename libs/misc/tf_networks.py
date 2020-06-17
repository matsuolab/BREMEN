import tensorflow as tf
import numpy as np


def get_activation_func(activation_type):
    if activation_type == 'leaky_relu':
        activation_func = tf.nn.leaky_relu
    elif activation_type == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_type == 'relu':
        activation_func = tf.nn.relu
    else:
        raise ValueError(
            "Unsupported activation type: {}!".format(activation_type)
        )
    return activation_func


def normc_initializer(shape, seed=1234, stddev=1.0, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    out = npr.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)


def weight_variable(shape, name, init_method=None, dtype=tf.float32,
                    init_para=None, seed=1234, trainable=True):
    """ @brief:
            Initialize weights
        @input:
            shape: list of int, shape of the weights
            init_method: string, indicates initialization method
            init_para: a dictionary,
            init_val: if it is not None, it should be a tensor
        @output:
            var: a TensorFlow Variable
    """

    if init_method is None or init_method == 'zero':
        initializer = tf.zeros_initializer(shape, dtype=dtype)

    if init_method == "normc":
        var = normc_initializer(
            shape, stddev=init_para['stddev'],
            seed=seed, dtype=dtype
        )
        return tf.get_variable(initializer=var, name=name, trainable=trainable)

    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"], maxval=init_para["maxval"],
            seed=seed, dtype=dtype
        )

    elif init_method == "constant":
        initializer = tf.constant_initializer(
            value=init_para["val"], dtype=dtype
        )

    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=init_para['uniform'], seed=seed, dtype=dtype
        )

    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )

    else:
        raise ValueError("Unsupported initialization method!")

    var = tf.get_variable(initializer=initializer(shape), name=name, trainable=trainable)

    return var


class MLP(object):
    def __init__(self, dims, scope, activation, init_data, dtype=tf.float32):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer

        self._activation = activation
        self._init_data = init_data

        # initialize variables
        with tf.variable_scope(scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='w',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype
                    )

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='b',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype
                    )

    def __call__(self, input_vec, reuse=False):
        self._h = [None] * self._num_layer
        self._input = input_vec

        with tf.variable_scope(self._scope, reuse=reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    layer = input_vec if ii == 0 else self._h[ii - 1]
                    self._h[ii] = tf.matmul(layer, self._w[ii]) + self._b[ii]

                    if self._activation[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

        return self._h[-1]
