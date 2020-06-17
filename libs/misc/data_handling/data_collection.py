import numpy as np


class DataCollection(object):
    def __init__(self, batch_size, max_size, shuffle=False, rng=None):
        self.p = 0
        self.data = None
        self.true_data = None
        self.inds = None
        self.batch_size = batch_size
        self.max_size = max_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(1) if rng is None else rng
        self.total_samples = 0

    def add_data(self, new_data, discard_ratio=0.0):
        num_new_data = new_data["rewards"].shape[0]
        if self.data is not None:
            num_data = int(len(self) * (1 - discard_ratio))
            start_idx = int(num_new_data * discard_ratio)
            if num_data + num_new_data > self.max_size:
                start_idx += int(num_new_data + num_data - self.max_size)

            self.data["actions"] = np.concatenate([
                self.data["actions"][start_idx:],
                new_data["actions"]
            ])
            self.data["next_observations"] = np.concatenate([
                self.data["next_observations"][start_idx:],
                new_data["next_observations"]
            ])
            self.data["observations"] = np.concatenate([self.data["observations"][start_idx:], new_data["observations"]])
            self.data["rewards"] = np.concatenate([self.data["rewards"][start_idx:], new_data["rewards"]])
        else:
            if num_new_data > self.max_size:
                self.data = {k: v[-self.max_size:] for k, v in new_data.items()}
            else:
                self.data = new_data

        self.inds = list(range(self.data["rewards"].shape[0]))
        self.total_samples += len(new_data["rewards"])

    def get_data(self):
        return self.data

    def get_total_samples(self):
        return self.total_samples

    def reset(self):
        self.p = 0

    def persist_true_data(self):
        self.true_data = self.data.copy()

    def restore_true_data(self):
        self.data = self.true_data.copy()

    def replace_data(self, new_data):
        self.data = None
        self.reset()
        self.add_data(new_data)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return self.data["rewards"].shape[0]

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None:
            n = np.minimum(self.batch_size, self.__len__())

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            self.inds = self.rng.permutation(self.__len__())
            for k in self.data.keys():
                self.data[k] = self.data[k][self.inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.__len__():
            self.reset()
            raise StopIteration

        # on intermediate iterations fetch the next batch
        obs = self.data["observations"][self.p: self.p + n]
        rew = self.data["rewards"][self.p: self.p + n]
        action = self.data["actions"][self.p: self.p + n]
        next_obs = self.data["next_observations"][self.p: self.p + n]
        self.p += self.batch_size

        return obs, action, next_obs, rew

    next = __next__
