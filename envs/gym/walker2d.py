from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/walker2d.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def _step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        if getattr(self, 'action_space', None):
            action = np.clip(
                action, self.action_space.low,
                self.action_space.high
            )

        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = ob[8]
        reward = reward_run + reward_ctrl + 1

        height = ob[0]
        ang = ob[1]
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -1e-3 * np.sum(np.square(acts), axis=1)
        reward_run = next_obs[:, 8]
        reward = reward_run + reward_ctrl + 1.0
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        """
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 0]
        # reward_height = -3.0 * tf.square(next_obs[:, 1] - 1.3)
        reward = reward_run + reward_ctrl
        return -reward
        """
        raise NotImplementedError

    def is_done(self, obs, next_obs):
        height = next_obs[0]
        ang = next_obs[1]
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        return done
