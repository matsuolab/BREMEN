import tensorflow as tf
import numpy as np
import logger
from model.dynamics import DynamicsModel


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
            for _ in range(self.n_layers):
                out = tf.layers.dense(out, self.size, activation=self.activation)
            out = tf.layers.dense(out, self.output_size, activation=self.output_activation)
        return out


class PNNDynamicsModel(DynamicsModel):
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
            scope="dynamics"):
        """ Note: Be careful about normalization """
        # Store arguments for later.
        self.env = env
        self.normalization = normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.val = val
        self.sess = sess
        self.scope = scope
        self.adam_scope = "adam_" + self.scope

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)
        obs_dim = env.observation_space.shape[0]
        acts_dim = env.action_space.shape[0]

        self.obs_dim = obs_dim
        self.acts_dim = acts_dim

        self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.acts_ph = tf.placeholder(tf.float32, shape=(None, acts_dim))
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))

        # Build NN.
        self.epsilon = 1e-10
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(self.obs_ph, self.acts_ph)

        # max_logvar and min_logvar
        with tf.variable_scope(self.scope):
            self.max_logvar = tf.get_variable('max_logvar', initializer=tf.ones(shape=[obs_dim])*0.5)
            self.min_logvar = tf.get_variable('min_logvar', initializer=-tf.ones(shape=[obs_dim])*10)

        self.mlp = mlp(
            output_size=obs_dim*2,
            scope=self.scope,
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation)
        cur_out = self.mlp(normalized_obs_and_acts)

        # adopt from https://arxiv.org/abs/1805.12114
        mean, logvar = self._get_mean_and_logvar_from_nn_output(cur_out)
        inv_var = tf.exp(-logvar)

        # enforce boundary consistence
        min_bounds, max_bounds = tf.minimum(self.max_logvar, self.min_logvar), tf.maximum(self.max_logvar, self.min_logvar)
        self.enforce_bound_consistency = tf.group(tf.assign(self.max_logvar, max_bounds), tf.assign(self.min_logvar, min_bounds))

        predicted_normalized_deltas = self._get_predicted_normalized_deltas_from_mean_and_logvar(mean, logvar)
        self.predicted_unnormalized_deltas = self._get_unnormalized_deltas(predicted_normalized_deltas)

        # Build cost function and optimizer.
        normalized_deltas = ((self.next_obs_ph - self.obs_ph) - normalization.mean_deltas) \
                            / (normalization.std_deltas + self.epsilon)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(mean - normalized_deltas) * inv_var, -1)) \
                    + tf.reduce_mean(tf.reduce_sum(logvar, -1)) + \
                    0.01 * (tf.reduce_sum(self.max_logvar) - tf.reduce_sum(self.min_logvar))
        self.loss_val = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=mean)

        with tf.variable_scope(self.adam_scope):
            self.update_op = tf.train.AdamOptimizer(learning_rate).\
                minimize(self.loss, var_list=tf.trainable_variables(self.scope))
        dyn_adam_vars = tf.trainable_variables(self.adam_scope)
        self.dyn_adam_init = tf.variables_initializer(dyn_adam_vars)

    def get_obs_dim(self):
        return self.obs_dim

    def fit(self, train_data, val_data):
        self.sess.run(self.dyn_adam_init)

        loss = 1000
        best_index = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                feed_dict = {
                    self.obs_ph: obs,
                    self.acts_ph: action,
                    self.next_obs_ph: next_obs,
                }
                self.sess.run([self.update_op, self.loss], feed_dict=feed_dict)
                self.sess.run(self.enforce_bound_consistency)

            if epoch % 5 == 0:
                loss_list = []
                for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                    feed_dict = {
                        self.obs_ph: obs,
                        self.acts_ph: action,
                        self.next_obs_ph: next_obs,
                    }
                    cur_loss = self.sess.run(self.loss_val, feed_dict=feed_dict)
                    loss_list.append(cur_loss)
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch
                logger.log(
                    "Dynamics optimization | epoch {}/{}: Loss = {}".
                    format(epoch, self.epochs, np.mean(loss_list))
                )

                if self.val:
                    if epoch - best_index >= 20 and epoch >= 50:
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

        return states + unnormalized_deltas

    def predict_tf(self, states, actions):
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(states, actions)
        nn_output = self.mlp(normalized_obs_and_acts, reuse=True)
        mean, logvar = self._get_mean_and_logvar_from_nn_output(nn_output)
        predicted_normalized_deltas = self._get_predicted_normalized_deltas_from_mean_and_logvar(mean, logvar)
        unnormalized_deltas = self._get_unnormalized_deltas(predicted_normalized_deltas)
        return states + unnormalized_deltas

    # ---- Private methods ----

    def _get_mean_and_logvar_from_nn_output(self, nn_output):
        mean = nn_output[:, :self.obs_dim]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - nn_output[:, self.obs_dim:])
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _get_normalized_obs_and_acts(self, obs, acts):
        normalized_obs = (obs - self.normalization.mean_obs) / (self.normalization.std_obs + self.epsilon)
        normalized_acts = (acts - self.normalization.mean_acts) / (self.normalization.std_acts + self.epsilon)
        return tf.concat([normalized_obs, normalized_acts], 1)

    def _get_predicted_normalized_deltas_from_mean_and_logvar(self, mean, logvar):
        rand = tf.random_normal(shape=tf.shape(mean), mean=0., stddev=1.)
        return mean + rand * tf.sqrt(tf.exp(logvar))

    def _get_unnormalized_deltas(self, normalized_deltas):
        return normalized_deltas * self.normalization.std_deltas + self.normalization.mean_deltas
