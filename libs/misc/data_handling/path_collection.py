import numpy as np
from libs.misc.data_handling.path import Path


class PathCollection:
    def __init__(self, paths=None):
        self.paths = []
        if paths is not None:
            self.add_paths(paths)

    def add_paths(self, new_paths):
        for path in new_paths:
            if isinstance(path, Path):
                self.paths.append(path)
            else:
                self.paths.append(Path(path))

        for path in new_paths:
            path.persist_true_observations()

    def restore_true_observations(self):
        for path in self.paths:
            path.restore_true_observations()

    def random_swap_for_predicted_prev_states(self, ratio, dynamics):
        for path in self.paths:
            path.random_swap_for_predicted_prev_states(ratio, dynamics)

    def convert_all_paths_to_data_collections(self):
        return PathCollection.to_data_collections(self.paths)

    @staticmethod
    def get_path_data_dict(paths):
        obs_dim = paths[0].obs_dim
        act_dim = paths[0].act_dim
        next_obs_dim = paths[0].next_obs_dim
        path_data = {
            "observations": np.empty([0, obs_dim]),
            "actions": np.empty([0, act_dim]),
            "next_observations": np.empty([0, next_obs_dim]),
            "rewards": np.array([])
        }
        for path in paths:
            observations, actions, next_observations, rewards = \
                path.get_path_data()

            path_data["observations"] = np.concatenate([path_data["observations"], observations])
            path_data["actions"] = np.concatenate([path_data["actions"], actions])
            path_data["next_observations"] = np.concatenate([path_data["next_observations"],next_observations])
            path_data["rewards"] = np.concatenate([path_data["rewards"], rewards])
        return path_data

    @staticmethod
    def to_data_collections(paths, split_ratio=0.666667):
        path_data = PathCollection.get_path_data_dict(paths)

        num_data = path_data["observations"].shape[0]
        _n = int(num_data * split_ratio)
        train_data = {
            "observations": path_data["observations"][:_n],
            "actions": path_data["actions"][:_n],
            "next_observations": path_data["next_observations"][:_n],
            "rewards": path_data["rewards"][:_n]
        }
        val_data = {
            "observations": path_data["observations"][_n:],
            "actions": path_data["actions"][_n:],
            "next_observations": path_data["next_observations"][_n:],
            "rewards": path_data["rewards"][_n:]
        }
        return train_data, val_data
