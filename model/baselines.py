import numpy as np
import tensorflow as tf
import os
import pickle
import logger
from libs.misc import tf_networks, misc_utils


class LinearFeatureBaseline(object):
    def __init__(self, name, observation_space=None, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.running_stats = None

    def _features(self, path):
        obs = path["observations"]
        normalized_obs = self.running_stats.apply_norm_np(obs)
        o = np.clip(normalized_obs, -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        value_targets = np.concatenate([path["value_targets"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(value_targets)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def pre_train(self, obses, rewards, gamma=0.99, gae=0.95):
        # self._features()
        normalized_obs = self.running_stats.apply_norm_np(obses)
        o = np.clip(normalized_obs, -10, 10)
        l = len(rewards)
        al = np.arange(l).reshape(-1, 1) / 100.0
        featmat = np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)
        # self.predict()
        if self._coeffs is None:
            baselines = np.zeros(len(rewards))
        else:
            baselines = featmat.dot(self._coeffs)
        baselines = np.append(baselines, 0)
        # compute gae & value target
        deltas = rewards + gamma * baselines[1:] - baselines[:-1]
        advantages = misc_utils.discount_cumsum(deltas, gamma * gae)
        value_targets = advantages + baselines[:-1]
        # self.fit()
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(value_targets)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)

    def add_running_stats(self, running_stats):
        if self.running_stats is not None:
            raise ValueError
        else:
            self.running_stats = running_stats

    def save_value_function(self, save_path, itr):
        with open(os.path.join(save_path, 'value_function_{}.pickle'.format(itr)), 'wb') as f:
            pickle.dump(self._coeffs, f)

    def restore_value_function(self, save_path, itr):
        with open(os.path.join(save_path, 'value_function_{}.pickle'.format(itr)), 'rb') as f:
            self._coeffs = pickle.load(f)


class MLPBaseline(object):
    def __init__(
        self,
        name,
        observation_space,
        epoch=20,
        network_shape=(200, 200),
        activation='tanh',
        sess=None
    ):

        self.epoch = epoch
        self.sess = sess
        self.obs_dim = obs_dim = observation_space.flat_dim
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim])
        self.target_ph = tf.placeholder(tf.float32, [None, 1])
        self.running_stats = None

        network_shape = [obs_dim] + list(network_shape) + [1]
        num_layer = len(network_shape) - 1
        act_type = [activation] * (num_layer - 1) + [None]

        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {
                    'w_init_method': 'normc',
                    'w_init_para': {'stddev': 1.0},
                    'b_init_method': 'constant',
                    'b_init_para': {'val': 0.0}
                }
            )
        self.baseline_mlp = tf_networks.MLP(
            dims=network_shape, scope=name,
            activation=act_type, init_data=init_data
        )

        self.value = self.baseline_mlp(self.obs_ph)
        self.value_loss = tf.losses.mean_squared_error(
            labels=self.target_ph,
            predictions=self.value
        )
        self.train_op = \
            tf.train.AdamOptimizer(
                3e-4, beta1=0.5, beta2=0.99, epsilon=1e-4).minimize(
                    self.value_loss, var_list=tf.trainable_variables(name))

    def fit(self, paths):
        obs = np.concatenate([p["observations"] for p in paths])
        value_targets = np.concatenate([path["value_targets"] for path in paths])
        value_targets = np.expand_dims(value_targets, 1)

        normalized_obs = self.running_stats.apply_norm_np(obs)
        for ep in range(self.epoch):
            loss, _ = self.sess.run(
                [self.value_loss, self.train_op],
                feed_dict={self.obs_ph: normalized_obs, self.target_ph: value_targets}
            )
            logger.log("value iteration: {} | loss: {}".format(ep, loss))

    def predict(self, path):
        obs = path["observations"]
        normalized_obs = self.running_stats.apply_norm_np(obs)
        return self.sess.run(self.value, feed_dict={self.obs_ph: normalized_obs})

    def add_running_stats(self, running_stats):
        if self.running_stats is not None:
            raise ValueError
        else:
            self.running_stats = running_stats
