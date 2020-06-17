from libs.misc.utils import get_inner_env
from libs.misc.visualization import turn_off_video_recording, turn_on_video_recording
from model.controllers import RandomController
from libs.misc.data_handling.path import Path
import logger
import tensorflow as tf
import numpy as np


class RolloutSampler:
    def __init__(self, env, dynamics=None, controller=None):
        self.env = env
        self.inner_env = get_inner_env(self.env)
        self.dynamics = dynamics
        self.controller = controller
        self.random_controller = RandomController(self.env)

    def update_dynamics(self, new_dynamics):
        self.dynamics = new_dynamics

    def update_controller(self, new_controller):
        self.controller = new_controller

    def generate_random_rollouts(self, num_paths, horizon=1000):
        logger.info("Generating random rollouts.")
        random_paths = self.sample(
            num_paths=num_paths,
            horizon=horizon,
            use_random_controller=True
        )
        logger.info("Done generating random rollouts.")
        return random_paths

    def generate_offline_data(self, data_file, n_train=int(1e6), horizon=1000):
        # datafile: str
        s1 = tf.train.load_variable(data_file, 'data/_s1/.ATTRIBUTES/VARIABLE_VALUE')
        s2 = tf.train.load_variable(data_file, 'data/_s2/.ATTRIBUTES/VARIABLE_VALUE')
        a1 = tf.train.load_variable(data_file, 'data/_a1/.ATTRIBUTES/VARIABLE_VALUE')
        r = tf.train.load_variable(data_file, 'data/_reward/.ATTRIBUTES/VARIABLE_VALUE')
        data_size = max(s1.shape[0], s2.shape[0], a1.shape[0], r.shape[0])
        n_train = min(n_train, data_size)
        paths = []
        for i in range(int(n_train/horizon)):
            path = Path()
            if i*horizon % 10000 == 0:
                print(i*horizon)
            for j in range(i*horizon, (i+1)*horizon, 1):
                obs = s1[j].tolist()
                action = a1[j].tolist()
                next_obs = s2[j].tolist()
                reward = r[j].tolist()
                path.add_timestep(obs, action, next_obs, reward)
            paths.append(path)

        return paths

    def sample(
            self,
            num_paths=3,
            horizon=1000,
            visualize=False,
            visualize_path_no=None,
            use_random_controller=False,
            evaluation=False,
            eval_model=False):
        paths = []
        rollouts = []
        residuals = []
        total_timesteps = 0
        path_num = 0

        controller = self._get_controller(use_random_controller=use_random_controller)

        while True:
            turn_off_video_recording()
            if visualize and not isinstance(controller, RandomController):
                if (visualize_path_no is None) or (path_num == visualize_path_no):
                    turn_on_video_recording()

            self._reset_env_for_visualization()

            logger.info("Path {} | total_timesteps {}.".format(path_num, total_timesteps))

            # update randomness
            if controller.__class__.__name__ == "MPCcontroller" \
                and hasattr(controller.dyn_model, 'update_randomness'):
                controller.dyn_model.update_randomness()

            path, rollout, residual, total_timesteps = self._rollout_single_path(horizon, controller, total_timesteps, evaluation, eval_model)
            paths.append(path)
            rollouts.append(rollout)
            residuals.append(residual)
            path_num += 1
            if total_timesteps >= num_paths * horizon:
                break

        turn_off_video_recording()

        if eval_model:
            return paths, rollouts, residuals
        else:
            return paths

    # ---- Private methods ----

    def _reset_env_for_visualization(self):
        if hasattr(self.env.wrapped_env, "stats_recorder"):
            setattr(self.env.wrapped_env.stats_recorder, "done", None)

    def _get_controller(self, use_random_controller=False):
        if use_random_controller:
            return self.random_controller
        return self.controller

    def _rollout_single_path(self, horizon, controller, total_timesteps, evaluation=False, eval_model=False):
        path = Path()
        rollout = Path() if eval_model else None
        residual = Path() if eval_model else None
        obs = self.env.reset()
        obs_img = np.array(obs.copy()).reshape(-1, len(obs))
        for horizon_num in range(1, horizon + 1):
            action = controller.get_action(obs, evaluation=evaluation)
            next_obs, reward, done, _info = self.env.step(action)
            path.add_timestep(obs, action, next_obs, reward)
            if eval_model:
                # step-wise evaluation
                _obs = np.array(obs).reshape(-1, len(obs))
                action = np.array(action).reshape(-1, len(action))
                next_obs_step_wise = self.dynamics.predict(_obs, action)
                rewards_step_wise = - self.inner_env.env.cost_np_vec(_obs, action, next_obs_step_wise)
                _next_obs = np.array(next_obs).reshape(-1, len(next_obs))
                deviation = next_obs_step_wise - _next_obs
                residual.add_timestep(deviation, action, next_obs_step_wise, rewards_step_wise)
                # trajectory-wise evaluation
                action = controller.get_action(obs_img, evaluation=evaluation)
                action = np.array(action).reshape(-1, len(action))
                next_obs_img = self.dynamics.predict(obs_img, action)
                rewards_img = - self.inner_env.env.cost_np_vec(obs_img, action, next_obs_img)
                rollout.add_timestep(obs_img, action, next_obs_img, rewards_img)
                obs_img = next_obs_img

            obs = next_obs
            if done or horizon_num == horizon:
                total_timesteps += horizon_num
                break
        return path, rollout, residual, total_timesteps

    def _load_variable_from_ckpt(ckpt_name, var_name):
        var_name_ = '/'.join(var_name.split('.')) + '/.ATTRIBUTES/VARIABLE_VALUE'
        return tf.train.load_variable(ckpt_name, var_name_)

    def _shuffle_indices_with_steps(n, steps=1, rand=None):
        """Randomly shuffling indices while keeping segments."""
        if steps == 0:
            return np.arange(n)
        if rand is None:
            rand = np.random
        n_segments = int(n // steps)
        n_effective = n_segments * steps
        batch_indices = rand.permutation(n_segments)
        batches = np.arange(n_effective).reshape([n_segments, steps])
        shuffled_batches = batches[batch_indices]
        shuffled_indices = np.arange(n)
        shuffled_indices[:n_effective] = shuffled_batches.reshape([-1])
        return shuffled_indices
