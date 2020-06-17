import pickle
import numpy as np
import itertools
import time
import random

import logger
from libs.misc import tensor_utils, misc_utils
from envs.vec_env import VecEnvExecutor


class VectorizedSampler(object):

    def __init__(self, algo, n_envs=None, offline_dataset=None, use_s_t=False, use_s_0=False):
        self.algo = algo
        self.n_envs = n_envs
        self.offline_dataset = offline_dataset
        self.use_s_t = use_s_t
        self.use_s_0 = use_s_0

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 1000))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, dynamics=None):
        logger.info("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        running_paths = [None] * self.vec_env.num_envs

        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        behavior_policy = self.algo.behavior_policy
        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            actions, agent_infos = policy.get_actions(obses)
            _,  behavior_policy_agent_infos = behavior_policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            behavior_policy_agent_infos = tensor_utils.split_tensor_dict_list(behavior_policy_agent_infos)
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if behavior_policy_agent_infos is None:
                behavior_policy_agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, obs, action, reward, agent_info, behavior_policy_agent_info, done in\
                    zip(itertools.count(), obses, actions, rewards, agent_infos, behavior_policy_agent_infos, dones):

                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        agent_infos=[],
                        behavior_policy_agent_infos=[],
                    )
                running_paths[idx]["observations"].append(obs)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["agent_infos"].append(agent_info)
                running_paths[idx]["behavior_policy_agent_infos"].append(behavior_policy_agent_info)
                if done:
                    paths.append(dict(
                        observations=self.algo.env.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.algo.env.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                        behavior_policy_agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["behavior_policy_agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            obses = next_obses

        print("PolicyExecTime: {}".format(policy_time) + "[sec]")
        print("EnvExecTime: {}".format(env_time) + "[sec]")
        print("ProcessExecTime: {}".format(process_time) + "[sec]")

        return paths, n_samples

    def update_stats(self, paths):
        obs = [path["observations"] for path in paths]
        obs = np.concatenate(obs)
        self.algo.running_stats.update_stats(np.reshape(obs, [-1, self.algo.obs_dim]))

    def process_samples(self, itr, paths):

        start = time.time()
        all_path_baselines = [self.algo.baseline.predict(path) for path in paths]
        print("baseline predicting time: {}".format(time.time()-start) + "[sec]")
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                self.algo.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["behavior_policy_kl"] = misc_utils.gauss_KL(
                path["agent_infos"]["mean"],
                path["agent_infos"]["logstd"],
                path["behavior_policy_agent_infos"]["mean"],
                path["behavior_policy_agent_infos"]["logstd"], axis=1)
            deltas = deltas - self.algo.alpha * path["behavior_policy_kl"]
            # GAE calculation
            path["advantages"] = misc_utils.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)

            # a trick to reduce variance but gives biased gradient
            path["value_targets"] = path["advantages"] + np.array(path_baselines[:-1])

        max_path_length = max([len(path["advantages"]) for path in paths])

        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        if self.algo.center_adv:
            raw_adv = np.concatenate([path["advantages"] for path in paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
        else:
            adv = [path["advantages"] for path in paths]

        adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        samples_data = dict(
            observations=obs,
            actions=actions,
            advantages=adv,
            paths=paths,
        )

        advantages = [path["advantages"] for path in paths]
        advantages = tensor_utils.pad_tensor_n(advantages, max_path_length)
        behavior_policy_kls = [path["behavior_policy_kl"] for path in paths]
        behavior_policy_kls = tensor_utils.pad_tensor_n(behavior_policy_kls, max_path_length)

        logger.record_tabular('advantages_mean', np.mean(advantages))
        logger.record_tabular('advantages_max', np.max(advantages))
        logger.record_tabular('advantages_min', np.min(advantages))
        logger.record_tabular('behavior_policy_kl_mean', np.mean(behavior_policy_kls))
        logger.record_tabular('behavior_policy_kl_max', np.max(behavior_policy_kls))
        logger.record_tabular('behavior_policy_kl_min', np.min(behavior_policy_kls))
        logger.dump_tabular()

        return samples_data
