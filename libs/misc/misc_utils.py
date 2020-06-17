import numpy as np
import tensorflow as tf
import scipy
import scipy.signal


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)  # the surrogate loss
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def gauss_KL(mu1, logstd1, mu2, logstd2, axis=None):
    # KL divergence between two paramaterized guassian distributions
    var1 = np.exp(2 * logstd1)
    var2 = np.exp(2 * logstd2)

    kl = np.sum(
        logstd2 - logstd1 + (var1 + np.square(mu1 - mu2)) / (2 * var2) - 0.5,
        axis=axis
    )
    return kl


class RunningStats(object):
    def __init__(self, shape, sess):
        self.sum = np.zeros(shape)
        self.square_sum = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.std = np.ones(shape)
        self.num_data = 0
        self.sess = sess

        self.mean_tf = tf.get_variable('stats_mean', shape=shape, initializer=tf.zeros_initializer, trainable=False)
        self.std_tf = tf.get_variable('stats_std', shape=shape, initializer=tf.ones_initializer, trainable=False)

        self.mean_tf_ph = tf.placeholder(tf.float32, shape=shape, name='stats_mean_ph')
        self.std_tf_ph = tf.placeholder(tf.float32, shape=shape, name='stats_std_ph')

        self.update_mean_op = self.mean_tf.assign(self.mean_tf_ph)
        self.update_std_op = self.std_tf.assign(self.std_tf_ph)

    def update_stats(self, data):
        self.sum += data.sum(axis=0)
        self.square_sum += np.square(data).sum(axis=0)
        self.num_data += np.shape(data)[0]

        self.mean = self.sum / self.num_data
        var = np.maximum(self.square_sum / self.num_data - np.square(self.mean), 1e-2)
        self.std = (var + 1e-6) ** 0.5

        self.sess.run(
            [self.update_mean_op, self.update_std_op],
            {self.mean_tf_ph: self.mean, self.std_tf_ph: self.std}
        )

    # for batched data
    def get_stats(self):
        return {"sum": self.sum, "square_sum": self.square_sum, "mean": self.mean, "std": self.std,
                "num_data": self.num_data}

    def insert_stats(self, stats):
        self.sum = stats["sum"]
        self.square_sum = stats["square_sum"]
        self.mean = stats["mean"]
        self.std = stats["std"]
        self.num_data = stats["num_data"]
        self.sess.run(
            [self.update_mean_op, self.update_std_op],
            {self.mean_tf_ph: self.mean, self.std_tf_ph: self.std}
        )

    def apply_norm_np(self, data):
        return (data - self.mean) / self.std

    def apply_norm_tf(self, data):
        return (data - self.mean_tf) / self.std_tf
