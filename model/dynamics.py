import tensorflow as tf
import numpy as np
import logger


class mlp(object):
    def __init__(
            self,
            output_size,
            scope='dynamics',
            n_layers=2,
            size=1000,
            activation=tf.nn.relu,
            output_activation=None):
        self.output_size = output_size
        self.scope = scope
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, input, reuse=False):
        out = input
        with tf.variable_scope(self.scope, reuse=reuse):
            l2_loss = 0.0
            for layer_i in range(self.n_layers):
                layer_name = "dense_{}".format(layer_i)
                out = tf.layers.dense(out, self.size, activation=self.activation, name=layer_name)
                with tf.variable_scope(layer_name, reuse=True):
                    weight = tf.get_variable("kernel")
                    l2_loss += tf.nn.l2_loss(weight)
            out = tf.layers.dense(out, self.output_size, activation=self.output_activation)
        return out, l2_loss


class Dynamics(object):
    def __init__(self):
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_intrinsic_rewards_only(self):
        logger.info("Pre-training enabled. Using only intrinsic reward.")
        self.int_rewards_only = True

    def combine_int_and_ext_rewards(self):
        logger.info("Using a combination of external and intrinsic reward.")
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_external_rewards_only(self):
        logger.info("Using external reward only.")
        self.int_rewards_only = False
        self.ext_rewards_only = True

    def information_gain(self, obses, acts, next_obses):
        return np.zeros([len(obses),])

    def information_gain_tf(self, obs, act, next_obs):
        raise NotImplementedError

    def process_rewards(self, ext_rewards, obses, actions, next_obses):
        if self.ext_rewards_only:
            return ext_rewards
        else:
            weighted_intrinsic_reward = self.information_gain(obses, actions, next_obses)
            if self.int_rewards_only:
                return weighted_intrinsic_reward
            else:
                return ext_rewards + weighted_intrinsic_reward


class DynamicsModel(Dynamics):
    def __init__(self, env, normalization, batch_size, epochs, val, sess):
        super().__init__()
        self.env = env
        self.normalization = normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.val = val
        self.sess = sess

        self.obs_dim = env.observation_space.shape[0]
        self.acts_dim = env.action_space.shape[0]
        self.mlp = None

        self.epsilon = 1e-10

    def get_obs_dim(self):
        raise NotImplementedError

    def update_randomness(self):
        pass

    def update_normalization(self, new_normalization):
        self.normalization = new_normalization

    def _build_placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        self.acts_ph = tf.placeholder(tf.float32, shape=(None, self.acts_dim))
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

    def _get_feed_dict(self, obs, action, next_obs):
        feed_dict = {
            self.obs_ph: obs,
            self.acts_ph: action,
            self.next_obs_ph: next_obs
        }
        return feed_dict

    def _get_normalized_obs_and_acts(self, obs, acts):
        normalized_obs = (obs[:, :self.obs_dim] - self.normalization.mean_obs) / (self.normalization.std_obs + self.epsilon)
        normalized_obs = tf.concat([normalized_obs, obs[:, self.obs_dim:]], axis=1)
        normalized_acts = (acts - self.normalization.mean_acts) / (self.normalization.std_acts + self.epsilon)
        return tf.concat([normalized_obs, normalized_acts], 1)

    def _get_predicted_normalized_deltas(self, states, actions):
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(states, actions)
        predicted_normalized_deltas, _ = self.mlp(normalized_obs_and_acts, reuse=True)
        return predicted_normalized_deltas

    def _get_unnormalized_deltas(self, normalized_deltas):
        return normalized_deltas * self.normalization.std_deltas + self.normalization.mean_deltas

    def _add_observations_to_unnormalized_deltas(self, states, unnormalized_deltas):
        return states[:, :self.obs_dim] + unnormalized_deltas

    def _get_normalized_deltas(self, deltas):
        return (deltas - self.normalization.mean_deltas) / (self.normalization.std_deltas + self.epsilon)


class NNDynamicsModel(DynamicsModel):
    def __init__(
            self,
            env,
            n_layers,
            size,
            activation,
            output_activation,
            normalization,
            batch_size,
            epochs,
            learning_rate,
            val,
            sess,
            scope="dynamics",
            reg_coeff=None,
            controller=None):
        super().__init__(env, normalization, batch_size, epochs, val, sess)
        self.scope = scope
        self.adam_scope = "adam_" + self.scope
        self.controller = controller
        if reg_coeff is None:
            self.reg_coeff = 1.0
        else:
            self.reg_coeff = reg_coeff

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)

        self._build_placeholders()

        # Build NN.
        with tf.variable_scope(self.scope):
            self.coeff = tf.get_variable('coeff', initializer=tf.constant(0.001), trainable=False)

        self.mlp = mlp(
            output_size=self.obs_dim,
            scope=self.scope,
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation)

        # Build cost function and optimizer.
        mse, l2_loss, self.predicted_unnormalized_deltas = self._get_loss()

        self.loss = mse + l2_loss * self.coeff * self.reg_coeff
        self.loss_val = mse
        with tf.variable_scope(self.adam_scope):
            self.update_op = tf.train.AdamOptimizer(learning_rate). \
                minimize(self.loss, var_list=tf.trainable_variables(self.scope))
        dyn_adam_vars = tf.trainable_variables(self.adam_scope)
        self.dyn_adam_init = tf.variables_initializer(dyn_adam_vars)

    def get_obs_dim(self):
        return self.obs_dim

    def fit(self, train_data, val_data):
        self.sess.run(self.dyn_adam_init)
        self.sess.run(tf.assign(self.coeff, 1. / len(train_data)))

        loss = 1000
        best_index = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                feed_dict = self._get_feed_dict(obs, action, next_obs)
                self.sess.run([self.update_op, self.loss], feed_dict=feed_dict)

            if epoch % 5 == 0:
                loss_list = []
                for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                    feed_dict = self._get_feed_dict(obs, action, next_obs)

                    cur_loss = self.sess.run(self.loss_val, feed_dict=feed_dict)
                    loss_list.append(cur_loss)
                logger.info("Validation loss = {}".format(np.mean(loss_list)))
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch

                if self.val:
                    if epoch - best_index >= 20:
                        break

    def predict(self, states, actions):
        assert(len(states) == len(actions))
        feed_dict = {
            self.obs_ph: states,
            self.acts_ph: actions,
        }
        unnormalized_deltas = self.sess.run(
            self.predicted_unnormalized_deltas,
            feed_dict=feed_dict)
        return np.array(states)[:, :self.obs_dim] + unnormalized_deltas

    def predict_tf(self, states, actions):
        return self._add_observations_to_unnormalized_deltas(
            states, self._get_unnormalized_deltas(self._get_predicted_normalized_deltas(states, actions)))

    def _get_loss(self):
        deltas = self.next_obs_ph - self.obs_ph
        labels = self._get_normalized_deltas(deltas)
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(self.obs_ph, self.acts_ph)
        predicted_normalized_deltas, l2_loss = self.mlp(normalized_obs_and_acts, reuse=tf.AUTO_REUSE)
        predicted_unnormalized_deltas = self._get_unnormalized_deltas(predicted_normalized_deltas)
        mse_loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=predicted_normalized_deltas)

        return mse_loss, l2_loss, predicted_unnormalized_deltas
