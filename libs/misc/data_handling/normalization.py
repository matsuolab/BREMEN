import numpy as np
import tensorflow as tf


class Normalization(object):
    def __init__(self, data, obs_dim=None, scope=None):
        if scope is None:
            self.scope = "normalization"
            self.reuse_scope = False
        else:
            self.scope = scope
            self.reuse_scope = tf.AUTO_REUSE

        observations, actions, next_observations = Normalization.get_data_components(data)
        if obs_dim is None:
            self.obs_dim = observations.shape[1]
        else:
            self.obs_dim = obs_dim

        effective_observations = observations[:, :self.obs_dim]
        effective_next_observations = next_observations[:, :self.obs_dim]
        with tf.variable_scope(self.scope, reuse=self.reuse_scope):
            mean_obs = np.mean(effective_observations, axis=0, dtype=np.float32)
            std_obs = np.std(effective_observations, axis=0, dtype=np.float32)
            mean_deltas = np.mean(effective_next_observations - effective_observations, axis=0, dtype=np.float32)
            std_deltas = np.std(effective_next_observations - effective_observations, axis=0, dtype=np.float32)
            mean_acts = np.mean(actions, axis=0, dtype=np.float32)
            std_acts = np.std(actions, axis=0, dtype=np.float32)

            self.mean_obs = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(mean_obs),
                name="mean_obs", trainable=False)
            self.std_obs = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(std_obs),
                name="std_obs", trainable=False)
            self.mean_deltas = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(mean_deltas),
                name="mean_deltas", trainable=False)
            self.std_deltas = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(std_deltas),
                name="std_deltas", trainable=False)
            self.mean_acts = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(mean_acts),
                name="mean_acts", trainable=False)
            self.std_acts = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(std_acts),
                name="std_acts", trainable=False)

            self.new_mean_obs = tf.placeholder(dtype=tf.float32, name='mean_obs_ph')
            self.new_std_obs = tf.placeholder(dtype=tf.float32, name='std_obs_ph')
            self.new_mean_deltas = tf.placeholder(dtype=tf.float32, name='mean_deltas_ph')
            self.new_std_deltas = tf.placeholder(dtype=tf.float32, name='std_deltas_ph')
            self.new_mean_acts = tf.placeholder(dtype=tf.float32, name='mean_acts_ph')
            self.new_std_acts = tf.placeholder(dtype=tf.float32, name='std_acts_ph')

            self.update_mean_obs = tf.assign(self.mean_obs, self.new_mean_obs)
            self.update_std_obs = tf.assign(self.std_obs, self.new_std_obs)
            self.update_mean_deltas = tf.assign(self.mean_deltas, self.new_mean_deltas)
            self.update_std_deltas = tf.assign(self.std_deltas, self.new_std_deltas)
            self.update_mean_acts = tf.assign(self.mean_acts, self.new_mean_acts)
            self.update_std_acts = tf.assign(self.std_acts, self.new_std_acts)

    def update(self, new_data):
        observations, actions, next_observations = Normalization.get_data_components(new_data)

        effective_observations = observations[:, :self.obs_dim]
        effective_next_observations = next_observations[:, :self.obs_dim]
        with tf.variable_scope(self.scope, reuse=self.reuse_scope):
            mean_obs = np.mean(effective_observations, axis=0, dtype=np.float32)
            std_obs = np.std(effective_observations, axis=0, dtype=np.float32)
            mean_deltas = np.mean(effective_next_observations - effective_observations, axis=0, dtype=np.float32)
            std_deltas = np.std(effective_next_observations - effective_observations, axis=0, dtype=np.float32)
            mean_acts = np.mean(actions, axis=0, dtype=np.float32)
            std_acts = np.std(actions, axis=0, dtype=np.float32)

            sess = tf.get_default_session()
            feed_dict = {
                self.new_mean_obs: mean_obs, self.new_std_obs: std_obs,
                self.new_mean_deltas: mean_deltas, self.new_std_deltas: std_deltas,
                self.new_mean_acts: mean_acts, self.new_std_acts: std_acts
            }

            sess.run(
                [
                    self.update_mean_obs, self.update_std_obs,
                    self.update_mean_deltas, self.update_std_deltas,
                    self.update_mean_acts, self.update_std_acts
                ],
                feed_dict=feed_dict
            )

    @staticmethod
    def get_data_components(data):
        return data["observations"], data["actions"], data["next_observations"]
