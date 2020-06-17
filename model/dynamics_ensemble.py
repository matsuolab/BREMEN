import numpy as np
import logger
import random
import time

from model.dynamics import Dynamics


class DynamicsModelEnsemble(Dynamics):
    def __init__(
            self, model_class, model_count, enable_particle_ensemble=False,
            particles=None, intrinsic_reward_coeff=None, obs_var=None
            ):
        super().__init__()
        self.model_class = model_class
        self.models = []
        self.model_count = model_count
        self.last_model_indices_used_for_prediction = None
        # particle dimension is represented by "tile number" of the first dimension of state / action
        # e.g. actions = np.tile(init_actions, (particles, 1))
        self.enable_particle_ensemble = enable_particle_ensemble
        self.particles = particles
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        self.obs_var = obs_var
        logger.info("Particle ensemble enabled? {}".format(self.enable_particle_ensemble))

    @property
    def env(self):
        return self.models[0].env

    def init_dynamic_models(self, **kwargs):
        for model_index in range(self.model_count):
            self.models.append(self.model_class(
                scope="dynamics_ensemble_model_{}".format(model_index),
                **kwargs))
        logger.info("An ensemble of {} dynamics model {} initialized".format(self.model_count, self.model_class))

    def get_model_obs_dim(self):
        if not self.models:
            raise ValueError("The models are not initialized yet!")
        return self.models[0].get_obs_dim()

    def predict(self, states, actions):
        model = self.models[self._generate_random_model_indices_for_prediction(num_models=1)[0]]
        return model.predict(states, actions)

    def predict_tf(self, states, actions):
        model = self.models[self._generate_random_model_indices_for_prediction(num_models=1)[0]]
        return model.predict_tf(states, actions)

    def fit(self, train_data, val_data):
        # fit all dynamics models
        for model_index in range(self.model_count):
            start = time.time()
            model = self.models[model_index]
            logger.info(
                "Fitting model {} (0-based) in the ensemble of {} models".format(model_index, self.model_count))
            model.fit(train_data, val_data)
            print("model fitting time: {}".format(time.time() - start) + "[sec]")

    def information_gain(self, states, actions, next_states):
        if not self.enable_particle_ensemble:
            return np.zeros([len(states), ])
        combined_pred_result = np.empty((0, len(states), self.get_model_obs_dim()))
        model_indices = self._generate_random_model_indices_for_prediction(num_models=self.particles)
        for particle_index in range(self.particles):
            model = self.models[model_indices[particle_index]]
            pred_result = model.predict(states, actions)
            combined_pred_result = np.append(combined_pred_result, [pred_result], axis=0)
        next_states_rep = np.tile(next_states, [self.particles, 1, 1])
        quad = np.sum((next_states_rep - combined_pred_result) ** 2, -1) / float(self.obs_var)
        quad = np.reshape(quad, [self.particles, -1])
        quad -= np.min(quad, 0)
        likelihood = np.exp(-quad)
        prob = likelihood / np.sum(likelihood, 0)
        entropy = - prob * np.log(prob + 1e-8)
        info_gain = np.log(float(self.particles)) - np.sum(entropy, 0)
        info_gain = np.nan_to_num(info_gain)
        return info_gain * self.intrinsic_reward_coeff

    def update_randomness(self):
        for model in self.models:
            model.update_randomness()

    def update_normalization(self, new_normalization):
        for model in self.models:
            model.update_normalization(new_normalization)

    # ======== Private methods ============

    def _generate_random_model_indices_for_prediction(self, num_models):
        model_indices = random.sample(range(self.model_count), k=num_models)
        self.last_model_indices_used_for_prediction = model_indices
        return model_indices
