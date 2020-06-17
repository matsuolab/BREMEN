from gym.core import Env
from gym.spaces import Box as GymBox
# from gym.wrappers.monitoring import Monitor
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf


class Box:

    def __init__(self, gym_box: GymBox):
        self.gym_box = gym_box

    @property
    def flat_dim(self):
        return np.prod(self.gym_box.shape)

    @property
    def shape(self):
        return self.gym_box.shape

    @property
    def dtype(self):
        return tf.float32

    @property
    def bounds(self):
        return self.gym_box.low, self.gym_box.high

    def flatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def sample(self):
        return self.gym_box.sample()

    def flatten(self, x):
        return np.asarray(x).flatten()

    def __repr__(self):
        return "Box Wrapper of shape {}".format(self.shape)

    def __eq__(self, other):
        return self.gym_box.__eq__(other)


class ProxyEnv(Env):

    def __init__(self, wrapped_env: Env):
        self._wrapped_env = wrapped_env
        self._wrapped_observation_space = Box(wrapped_env.observation_space)
        self._wrapped_action_space = Box(wrapped_env.action_space)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    @property
    def action_space(self):
        return self._wrapped_action_space

    @property
    def observation_space(self):
        return self._wrapped_observation_space

    def step(self, action, **kwargs):
        return self._wrapped_env.step(action, **kwargs)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if isinstance(self._wrapped_env, Monitor):
            self._wrapped_env._close()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)
