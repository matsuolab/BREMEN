import tensorflow as tf
import numpy as np
import time
import logger

from libs.misc import tf_utils, misc_utils
from libs.samplers.vectorized_sampler import VectorizedSampler


class TRPO(object):
    def __init__(
        self,
        env,
        inner_env,
        policy,
        baseline,
        sess,
        scope=None,
        behavior_policy=None,
        alpha=0.1,  # for scaling KL penalty in advantage
        n_itr=500,
        batch_size=5000,
        max_path_length=1000,
        center_adv=True,
        discount=0.99,
        gae_lambda=1.0,
        damping=0.1,
        cg_itrs=10,
        target_kl=0.01,
        offline_dataset=None,
        use_s_t=False,
        use_s_0=False,
    ):
        self.env = env
        self.inner_env = inner_env
        self.policy = policy
        self.baseline = baseline
        self.sess = sess
        if scope is None:
            self.scope = "trpo"
        else:
            self.scope = scope
        self.behavior_policy = behavior_policy
        self.alpha = alpha
        self.n_itr = n_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.center_adv = center_adv
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.damping = damping
        self.cg_itrs = cg_itrs
        self.target_kl = target_kl
        self.sampler = VectorizedSampler(self, None, offline_dataset, use_s_t, use_s_0)

        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        with tf.variable_scope(self.scope):
            # four useful functions
            self.flatten = tf_utils.flatten
            self.unflatten = tf_utils.unflatten(policy.trainable_variables)
            self.get_params = tf_utils.GetFlat(sess, policy.trainable_variables)
            self.set_params = tf_utils.SetFromFlat(sess, policy.trainable_variables)

            # keep running mean for observation
            self.running_stats = misc_utils.RunningStats([self.obs_dim], sess)
            self.policy.add_running_stats(self.running_stats)
            self.baseline.add_running_stats(self.running_stats)

            # build placeholder
            self._build_placeholder()

            # build computational graph
            self._init_opt()

    def _build_placeholder(self):
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim])
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim])
        self.old_act_dist_mean_ph = tf.placeholder(tf.float32, [None, self.act_dim])
        self.old_act_dist_logstd_ph = tf.placeholder(tf.float32, [None, self.act_dim])
        self.adv_ph = tf.placeholder(tf.float32, [None])

        self.flat_tangents = tf.placeholder(tf.float32, [None])

    def _init_opt(self):
        self.act_dist_mean, self.act_dist_logstd \
            = self.policy.get_dist_tf(self.obs_ph)

        # log_prob of actions
        self.log_p_act = tf_utils.gauss_log_prob(
            self.act_dist_mean,
            self.act_dist_logstd,
            self.act_ph
        )
        # log_prob of actions in old_dist
        self.log_oldp_act = tf_utils.gauss_log_prob(
            self.old_act_dist_mean_ph,
            self.old_act_dist_logstd_ph,
            self.act_ph
        )

        # compute the ratio
        self.ratio = tf.exp(self.log_p_act - self.log_oldp_act)

        # compute the kl divergence
        self.kl = tf_utils.gauss_KL(
            self.old_act_dist_mean_ph,
            self.old_act_dist_logstd_ph,
            self.act_dist_mean,
            self.act_dist_logstd
        ) / self.batch_size

        # compute the entropy
        self.entropy = tf_utils.gauss_ent(
            self.act_dist_mean,
            self.act_dist_logstd
        ) / self.batch_size

        # compute the surrogate loss and its gradient
        self.surr_loss = -tf.reduce_mean(
            self.ratio * self.adv_ph
        )

        self.surr_grad = tf_utils.flatgrad(
            self.surr_loss, self.policy.trainable_variables
        )

        # compute kl gradient
        kl_fixed = tf_utils.gauss_kl_fixed(
            self.act_dist_mean,
            self.act_dist_logstd
        ) / self.batch_size
        kl_grad = tf.gradients(
            kl_fixed, self.policy.trainable_variables
        )

        # get tangents
        tangents = self.unflatten(self.flat_tangents)
        kl_grad_times_tangents = [tf.reduce_sum(g * t) for (g, t) in zip(kl_grad, tangents)]

        self.fisher_vector_prod = \
            tf_utils.flatgrad(kl_grad_times_tangents, self.policy.trainable_variables)

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, dynamics=None):
        return self.sampler.obtain_samples(itr, dynamics=dynamics)

    def update_stats(self, paths):
        self.sampler.update_stats(paths)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def fit_baseline(self, paths):
        self.baseline.fit(paths)

    def pre_train_baseline(self, obses, rewards, gamma=0.99, gae=0.95):
        self.baseline.pre_train(obses, rewards, gamma, gae)

    def optimize_policy(self, itr, samples_data):
        # get old_dist
        feed_dict = {self.obs_ph: np.reshape(samples_data["observations"], [-1, self.obs_dim])[:self.batch_size]}
        old_act_dist_mean, old_act_dict_logstd = \
            self.sess.run([self.act_dist_mean, self.act_dist_logstd], feed_dict)

        # get old_params
        old_params = self.get_params()

        # get feed_dict

        feed_dict = {
            self.obs_ph: np.reshape(samples_data["observations"], [-1, self.obs_dim])[:self.batch_size],
            self.act_ph: np.reshape(samples_data["actions"], [-1, self.act_dim])[:self.batch_size],
            self.adv_ph: np.reshape(samples_data["advantages"], [-1])[:self.batch_size],  # GAE
            self.old_act_dist_mean_ph: old_act_dist_mean,
            self.old_act_dist_logstd_ph: old_act_dict_logstd
        }

        def fisher_vector_prod(vec):
            feed_dict[self.flat_tangents] = vec
            return self.sess.run(self.fisher_vector_prod, feed_dict) \
                + vec * self.damping

        def loss_fn(params):
            self.set_params(params)
            loss = self.sess.run(self.surr_loss, feed_dict)
            self.set_params(old_params)
            return loss

        # compute full_step using conjugate gradient
        surr_grad = self.sess.run(self.surr_grad, feed_dict)

        full_step = misc_utils.conjugate_gradient(
            fisher_vector_prod, -surr_grad, self.cg_itrs
        )

        # do line search
        vFv = 0.5 * full_step.dot(fisher_vector_prod(full_step))
        full_step = full_step / np.sqrt(vFv / self.target_kl)
        negative_g_dot_step = -surr_grad.dot(full_step)

        new_params = misc_utils.linesearch(
            loss_fn, old_params,
            full_step, negative_g_dot_step
        )

        self.set_params(new_params)

    def reinit_with_source_policy(self, source_policy):
        self.running_stats.insert_stats(source_policy.running_stats.get_stats())
