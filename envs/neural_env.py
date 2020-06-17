import numpy as np
import random


class NeuralNetEnv:

    def __init__(self, env, inner_env, dynamics, offline_dataset=None, use_s_t=False, use_s_0=False):
        self.vectorized = True
        self.env = env
        self.inner_env = inner_env
        # self.is_done = getattr(inner_env, 'is_done', lambda x, y: np.asarray([False] * len(x)))
        self.dynamics = dynamics
        self.offline_dataset = offline_dataset
        self.use_s_t = use_s_t
        self.use_s_0 = use_s_0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        if self.use_s_t:
            rnd_idx = random.randrange(len(self.offline_dataset))
            self.state = self.offline_dataset[rnd_idx]
        elif self.use_s_0:
            horizon = 1000
            rnd_idx = random.randrange(int(len(self.offline_dataset)/horizon))
            self.state = self.offline_dataset[rnd_idx*horizon]
        else:
            self.state = self.env.reset()
        observation = np.copy(self.state)
        return observation

    def step(self, action, use_states=None):
        action = np.clip(action, *self.action_space.bounds)
        if use_states is not None:
            next_observation = self.dynamics.predict([use_states], [action])[0]
            obs_dim = self.env.observation_space.shape[0]
            next_observation[:obs_dim] = np.clip(next_observation[:obs_dim], *self.observation_space.bounds)
            next_observation[:obs_dim] = np.clip(next_observation[:obs_dim], -1e5, 1e5)
        else:
            next_observation = self.dynamics.predict([self.state], [action])[0]
            next_observation = np.clip(next_observation, *self.observation_space.bounds)
            next_observation = np.clip(next_observation, -1e5, 1e5)

        if hasattr(self.inner_env, "env"):
            reward = - self.inner_env.env.cost_np_vec(self.state[None], action[None], np.array([next_observation]))[0]
        else:
            reward = - self.inner_env.cost_np_vec(self.state[None], action[None], np.array([next_observation]))[0]

        done = self.is_done(self.state[None], next_observation)[0]
        self.state = np.reshape(next_observation, -1)
        return self.inner_env.step({"next_obs": next_observation, "reward": reward, "done": done})

    def render(self):
        print('current state:', self.state)

    def vec_env_executor(self, n_envs, max_path_length):
        return VecSimpleEnv(env=self, inner_env=self.inner_env, n_envs=n_envs, max_path_length=max_path_length)

    def terminate(self):
        self.env.terminate()

    def is_done(self, obses, next_obses):
        dones = []
        for obs, next_obs in zip(obses, next_obses):
            dones.append(self.inner_env.env.is_done(obs, next_obs))
        return np.array(dones)


class VecSimpleEnv(object):

    def __init__(self, env, inner_env, n_envs, max_path_length):
        self.env = env
        self.inner_env = inner_env
        self.n_envs = n_envs
        self.num_envs = n_envs
        self.ts = np.zeros((self.n_envs,))
        self.max_path_length = max_path_length
        self.obs_dim = env.observation_space.shape[0]
        self.states = np.zeros((self.n_envs, self.obs_dim))

    def reset(self, dones=None):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        for i, done in enumerate(dones):
            if done:
                self.states[i] = self.env.reset()
        self.ts[dones] = 0
        return self.states[dones]

    def step(self, actions, use_states=None):
        self.ts += 1
        actions = np.clip(actions, *self.env.action_space.bounds)
        next_observations = self.get_next_observation(actions, use_states=use_states)
        if use_states is not None:
            obs_dim = self.env.observation_space.shape[0]
            next_observations[:, :obs_dim] = np.clip(next_observations[:, :obs_dim], *self.env.observation_space.bounds)
            next_observations[:, :obs_dim] = np.clip(next_observations[:, :obs_dim], -1e5, 1e5)
        else:
            next_observations = np.clip(next_observations, *self.env.observation_space.bounds)
            next_observations = np.clip(next_observations, -1e5, 1e5)
        if hasattr(self.env.inner_env, "cost_np_vec"):
            rewards = - self.env.inner_env.cost_np_vec(self.states, actions, next_observations)
        else:
            rewards = - self.env.inner_env.env.cost_np_vec(self.states, actions, next_observations)
        self.states = next_observations
        dones = self.env.is_done(self.states, next_observations)
        dones[self.ts >= self.max_path_length] = True
        if np.any(dones):
            self.reset(dones)
        return self.states, rewards, dones, dict()

    def get_next_observation(self, actions, use_states=None):
        if use_states is not None:
            return self.env.dynamics.predict(use_states, actions)
        return self.env.dynamics.predict(self.states, actions)

    def terminate(self):
        self.env.terminate()
