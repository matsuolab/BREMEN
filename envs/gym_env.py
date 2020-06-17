# import gym
# import gym.wrappers
# import gym.envs
# import gym.spaces
# import traceback
# import logging
#
# try:
#     from gym.wrappers.monitoring import logger as monitor_logger
#
#     monitor_logger.setLevel(logging.WARNING)
# except Exception as e:
#     traceback.print_exc()
#
# import os
# import numpy as np
# from rllab.misc import logger
#
#
# class CappedCubicVideoSchedule(object):
#     # Copied from gym, since this method is frequently moved around
#     def __call__(self, count):
#         if count < 1000:
#             return int(round(count ** (1. / 3))) ** 3 == count
#         else:
#             return count % 1000 == 0
#
#
# class FixedIntervalVideoSchedule(object):
#     def __init__(self, interval):
#         self.interval = interval
#
#     def __call__(self, count):
#         return count % self.interval == 0
#
#
# class NoVideoSchedule(object):
#     def __call__(self, count):
#         return False
#
#
# class GymEnv(object):
#     def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
#                  force_reset=False):
#         if log_dir is None:
#             if logger.get_snapshot_dir() is None:
#                 logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
#             else:
#                 log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
#
#         env = gym.make(env_name)
#         self.env = env
#         self.env_id = env.spec.id
#
#         assert not (not record_log and record_video)
#
#         if log_dir is None or record_log is False:
#             self.monitoring = False
#         else:
#             if not record_video:
#                 video_schedule = NoVideoSchedule()
#             else:
#                 if video_schedule is None:
#                     video_schedule = CappedCubicVideoSchedule()
#             self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
#             self.monitoring = True
#
#         self._observation_space = env.observation_space
#         self._action_space = env.action_space
#         self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
#         self._log_dir = log_dir
#         self._force_reset = force_reset
#
#         self.metadata = {'render.modes': ['human', 'rgb_array']}
#         self.reward_range = (-np.inf, np.inf)
#         self.unwrapped = self
#         self._configured = False
#         self.spec = None
#
#     @property
#     def inner_env(self):
#         env = self.env
#         while hasattr(env, "env"):
#             env = env.env
#         return env
#
#     @property
#     def observation_space(self):
#         return self._observation_space
#
#     @property
#     def action_space(self):
#         return self._action_space
#
#     @property
#     def horizon(self):
#         return self._horizon
#
#     def reset(self):
#         if self._force_reset and hasattr(self.env, 'stats_recorder'):
#             recorder = self.env.stats_recorder
#             if recorder is not None:
#                 recorder.done = True
#
#         return self.env.reset()
#
#     def step(self, action_or_predicted_result):
#         if isinstance(action_or_predicted_result, dict):
#             return self._step_with_predicted_dynamics(**action_or_predicted_result)
#         else:
#             next_obs, reward, done, info = self.env.step(action_or_predicted_result)
#             return next_obs, reward, done, info
#
#     def _step_with_predicted_dynamics(self, next_obs, reward, done):
#         qpos = self.inner_env.model.data.qpos.flatten()
#         qvel = self.inner_env.model.data.qvel.flatten()
#         self.env.env.set_state(qpos, qvel)
#         return next_obs, reward, done, {}
#
#     def cost_np_vec(self, obs, acts, next_obs):
#         return self.env.env.cost_np_vec(obs, acts, next_obs)
#
#     def cost_tf_vec(self, obs, acts, next_obs):
#         return self.env.env.cost_tf_vec(obs, acts, next_obs)
#
#     def render(self, **kwargs):
#         return self.env.render(**kwargs)
#
#     def terminate(self):
#         if self.monitoring:
#             self.env._close()
#             if self._log_dir is not None:
#                 print("""
#     ***************************
#     Training finished! You can upload results to OpenAI Gym by running the following command:
#     python scripts/submit_gym.py %s
#     ***************************
#                 """ % self._log_dir)
#
#     def get_geom_xpos(self):
#         return self.inner_env.data.geom_xpos
