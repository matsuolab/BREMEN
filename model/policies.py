import numpy as np
import tensorflow as tf
from libs.misc import tf_networks, misc_utils
import logger
import time

LOGSTD_MIN = -5
LOGSTD_MAX = 0


class MLPPolicy(object):
    def __init__(
        self,
        name,
        observation_space,
        action_space,
        init_logstd=0.0,
        network_shape=(64, 64),
        activation='tanh',
        sess=None,
        params=None
    ):
        self.name = name
        self.running_stats = None
        self.sess = sess
        self.params = params

        obs_dim = observation_space.flat_dim
        self.obs_dim = obs_dim
        act_dim = action_space.flat_dim
        self.act_dim = act_dim
        act_low, act_high = action_space.bounds
        self._act_mean = (act_high + act_low) / 2.0
        self._act_mags = (act_high - act_low) / 2.0

        network_shape = [obs_dim] + list(network_shape) + [act_dim*2]
        num_layer = len(network_shape) - 1
        self.num_layer = num_layer
        act_type = [activation] * (num_layer - 1) + [None]

        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {
                    'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                    'b_init_method': 'constant', 'b_init_para': {'val': 0.0}
                }
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std
        self.mean_network = tf_networks.MLP(
            dims=network_shape, scope=name,
            activation=act_type, init_data=init_data
        )

        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim])

        self.mean, self.logstd = self._process_mean_network_output(self.mean_network(self.obs_ph))
        with tf.variable_scope(self.name):
            self.init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

        # for behavior policy
        if self.name == 'behavior_policy':
            self.acts_ph = tf.placeholder(tf.float32, shape=(None, self.act_dim))
            self.running_stats = misc_utils.RunningStats(obs_dim, sess)
            self.bc_adam_scope = "adam_" + self.name
            self.bc_loss = self._get_bc_loss()
            with tf.variable_scope(self.bc_adam_scope):
                self.bc_update_op = tf.train.AdamOptimizer(self.params['behavior_policy']['learning_rate']).\
                    minimize(self.bc_loss, var_list=tf.trainable_variables(self.name))
            bc_adam_vars = tf.trainable_variables(self.bc_adam_scope)
            self.bc_adam_init = tf.variables_initializer(bc_adam_vars)

        self.gaussian_sigma = self.params['gaussian']

    @property
    def vectorized(self):
        return True

    @property
    def trainable_variables(self):
        return tf.trainable_variables(self.name)

    def initialize_variables(self):
        self.init_op.run()

    def get_action(self, ob):
        ob = np.reshape(ob, [1, -1])
        normlaized_ob = self.running_stats.apply_norm_np(ob)
        mean, logstd = self.sess.run(
            [self.mean, self.logstd],
            feed_dict={self.obs_ph: normlaized_ob}
        )
        rnd = np.random.normal(loc=0., scale=self.gaussian_sigma, size=mean.shape)
        if self.params['const_sampling']:
            action = rnd + mean
        else:
            action = rnd * np.exp(logstd) + mean
        return action[0], dict(mean=mean, logstd=logstd)

    def get_actions(self, obs):
        normlaized_obs = self.running_stats.apply_norm_np(obs)
        means, logstds = self.sess.run(
            [self.mean, self.logstd],
            feed_dict={self.obs_ph: normlaized_obs}
        )
        rnd = np.random.normal(loc=0., scale=self.gaussian_sigma, size=means.shape)
        if self.params['const_sampling']:
            actions = rnd + means
        else:
            actions = rnd * np.exp(logstds) + means
        return actions, dict(mean=means, logstd=logstds)

    def get_actions_tf(self, obs):
        normlaized_obs = self.running_stats.apply_norm_tf(obs)
        means = self.mean_network(normlaized_obs, reuse=True)
        rnd = tf.random_normal(shape=obs.shape)
        actions = rnd * tf.exp(self.logstd) + means
        return actions, None

    def get_dist_tf(self, obs):
        normlaized_obs = self.running_stats.apply_norm_tf(obs)
        return self._process_mean_network_output(self.mean_network(normlaized_obs))

    def add_running_stats(self, running_stats):
        if self.running_stats is not None:
            raise ValueError
        else:
            self.running_stats = running_stats

    def _process_mean_network_output(self, mean_network_output):
        mean, logstd = tf.split(mean_network_output, 2, axis=-1)
        mean = tf.tanh(mean) * self._act_mags + self._act_mean
        logstd = LOGSTD_MIN + 0.5 * (LOGSTD_MAX - LOGSTD_MIN) * (tf.tanh(logstd) + 1)
        return mean, logstd

    def _get_bc_loss(self):
        predicted_action = self.mean + tf.random_normal(tf.shape(self.acts_ph)) * tf.exp(self.logstd)
        mse_loss = tf.losses.mean_squared_error(
            labels=self.acts_ph,
            predictions=predicted_action)
        return mse_loss

    def fit_as_bc(self, train_data, val_data, rollout_sampler=None):
        total_steps = self.params['behavior_policy']['total_train_steps']
        itr = 1
        while True:
            for obs, action, _next_obs, _reward in train_data:
                # start = time.time()
                feed_dict = {
                    self.obs_ph: self.running_stats.apply_norm_np(obs),
                    self.acts_ph: action
                }
                _, train_loss = self.sess.run([self.bc_update_op, self.bc_loss], feed_dict=feed_dict)
                if itr % 10000 == 0:
                    logger.info("BC iter " + str(itr) + " train loss: " + str(train_loss))
                    # test in real env
                    if rollout_sampler:
                        rl_paths = rollout_sampler.sample(
                            num_paths=self.params['num_path_onpol'],
                            horizon=self.params['env_horizon'],
                            evaluation=True
                        )
                        returns = np.mean(np.array([sum(path["rewards"]) for path in rl_paths]))
                        logger.info("BC iter " + str(itr) + " average return: " + str(returns))
                if itr < total_steps:
                    itr += 1
                else:
                    return
