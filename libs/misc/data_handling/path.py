import random
from libs.misc.utils import get_inner_env


class Path:
    def __init__(self, path_dict: dict=None):
        if path_dict:
            self.observations = path_dict["observations"]
            self.actions = path_dict["actions"]
            self.next_observations = path_dict["next_observations"]
            self.rewards = path_dict["rewards"]
        else:
            self.observations = []
            self.actions = []
            self.next_observations = []
            self.rewards = []

        self.true_observations = []

    @property
    def length(self):
        return len(self.observations)

    @property
    def obs_dim(self):
        return len(self.observations[0])

    @property
    def act_dim(self):
        return len(self.actions[0])

    @property
    def next_obs_dim(self):
        return len(self.next_observations[0])

    def add_observation(self, obs):
        self.observations.append(obs)

    def add_action(self, action):
        self.actions.append(action)

    def add_next_observation(self, next_obs):
        self.next_observations.append(next_obs)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_timestep(self, obs, action, next_obs, reward):
        self.add_observation(obs)
        self.add_action(action)
        self.add_next_observation(next_obs)
        self.add_reward(reward)

    def persist_true_observations(self):
        self.true_observations = self.observations.copy()

    def restore_true_observations(self):
        self.observations = self.true_observations.copy()

    def random_swap_for_predicted_prev_states(self, ratio, dynamics):
        n_swaps = int((self.length - 1) * ratio)
        time_steps_to_swap = random.sample(range(1, self.length), n_swaps)
        for time_step in sorted(time_steps_to_swap, reverse=True):
            prev_state = self.observations[time_step - 1]
            prev_action = self.actions[time_step - 1]
            predicted_cur_state = dynamics.predict([prev_state], [prev_action])[0]
            self.observations[time_step] = predicted_cur_state

    def get_path_data(self):
        return self.observations, self.actions, self.next_observations, self.rewards

    def __getitem__(self, item):
        return getattr(self, item)
