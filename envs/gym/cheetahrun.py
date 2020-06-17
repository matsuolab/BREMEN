from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

_RUN_SPEED = 10


class CheetahRunEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=5):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/half_cheetah.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def _step(self, action):
        start_ob = self._get_obs()
        x = start_ob[8]
        lower = _RUN_SPEED
        upper = float('inf')
        margin = _RUN_SPEED
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        reward = np.where(in_bounds, 1.0, np.where(abs(d) < 1, 1 - d, 0.0))

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if getattr(self, 'action_space', None):
            action = np.clip(
                action, self.action_space.low,
                self.action_space.high
            )

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cost_np_vec(self, obs, acts, next_obs):
        x = obs[:, 8]
        lower = _RUN_SPEED
        upper = float('inf')
        margin = _RUN_SPEED
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        reward = np.where(in_bounds, 1.0, np.where(abs(d) < 1, 1 - d, 0.0))
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        raise NotImplementedError
        """
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 0]
        reward = reward_run + reward_ctrl
        return -reward
        """

    def is_done(self, obs, next_obs):
        done = False
        return done
