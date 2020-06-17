import argparse
import numpy as np
import tensorflow as tf

from libs.misc.utils import get_session, get_env
from libs.misc.saving_and_loading import confirm_restoring_policy, restore_policy_for_video
from libs.misc.initial_configs.policy_config import create_policy_from_params
from params_preprocessing import process_params
from libs.misc import misc_utils
from libs.misc.data_handling.utils import add_path_data_to_collection_and_update_normalization
from libs.misc.data_handling.data_collection import DataCollection
from libs.misc.data_handling.path_collection import PathCollection
from libs.misc.data_handling.rollout_sampler import RolloutSampler


def get_data_from_offline_batch(params, env, normalization_scope=None, model='dynamics', split_ratio=0.666667):
    train_collection = DataCollection(
        batch_size=params[model]['batch_size'],
        max_size=params['max_train_data'], shuffle=True)
    val_collection = DataCollection(batch_size=params[model]['batch_size'],
                                    max_size=params['max_val_data'],
                                    shuffle=False)
    rollout_sampler = RolloutSampler(env)
    rl_paths = rollout_sampler.generate_offline_data(
            data_file=params['data_file'],
            n_train=params["n_train"]
        )
    path_collection = PathCollection()
    obs_dim = env.observation_space.shape[0]
    normalization = add_path_data_to_collection_and_update_normalization(
        rl_paths, path_collection,
        train_collection, val_collection,
        normalization=None,
        split_ratio=split_ratio,
        obs_dim=obs_dim,
        normalization_scope=normalization_scope)
    return train_collection, val_collection, normalization, path_collection, rollout_sampler


def main(params, exp_name):
    sess = get_session(interactive=True)
    env = get_env(params['env_name'], params.get('video_dir'))

    train_collection, _, _, _, _ = \
        get_data_from_offline_batch(params, env, split_ratio=1.0)

    policy = create_policy_from_params(params, env, sess)
    policy.add_running_stats(misc_utils.RunningStats([env.observation_space.shape[0]], sess))
    policy_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
    policy_saver = tf.train.Saver(var_list=policy_variables)
    tf.global_variables_initializer().run()

    # restore TRPO policy
    if confirm_restoring_policy(params):
        restore_path = params['restore_path']
        restore_policy_for_video(restore_path, policy_saver, sess)
        print('policy is restored.')

    policy.running_stats.update_stats(train_collection.data["observations"])

    for i_episodes in range(params['num_path_onpol']):
        obs = env.reset()
        for t in range(params['env_horizon']):
            env.render()
            action, _ = policy.get_action(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiment options')
    parser.add_argument('--env')
    parser.add_argument('--exp_name')
    parser.add_argument('--sub_exp_name', default="")
    parser.add_argument('--noise', default='pure')
    parser.add_argument('--algo', default='trpo')
    parser.add_argument('--param_path', default=None)
    parser.add_argument('--interval', type=int, default=80)
    parser.add_argument('--onpol_iters', type=int, default=500)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--trpo_batch_size', type=int, default=50000)
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[1234, 4321, 2341, 3341, 789])
    parser.add_argument('--n_train', type=int, default=1000000)
    parser.add_argument('--alpha', type=float, default=0.1)  # hyperparam for scaling KL
    parser.add_argument('--target_kl', type=float, default=0.01)  # stepsize of TRPO
    parser.add_argument('--ensemble_model_count', type=int, default=5)  # number of dynamics model ensemble
    parser.add_argument('--param_value', action='store_true')  # if true, use parametric value function
    parser.add_argument('--save_variables', action='store_true')
    parser.add_argument('--restart_iter', type=int, default=0)
    parser.add_argument('--restore_bc_variables', action='store_true')
    parser.add_argument('--restore_policy_variables', action='store_true')
    parser.add_argument('--restore_dynamics_variables', action='store_true')
    parser.add_argument('--restore_offline_data', action='store_true')
    parser.add_argument('--restore_value', action='store_true')
    parser.add_argument('--bc_init', action='store_true')
    parser.add_argument('--use_s_t', action='store_true')
    parser.add_argument('--use_s_0', action='store_true')
    parser.add_argument('--pretrain_value', action='store_true')
    parser.add_argument('--video_dir', default=None)
    parser.add_argument('--restore_path', default=None)
    parser.add_argument('--gaussian', type=float, default=1.0)
    parser.add_argument('--const_sampling', action='store_true')
    parser.add_argument('--all_bc', action='store_true')
    parser.add_argument('--eval_model', action='store_true')
    options = parser.parse_args()
    if options.noise == 'pure':
        exp_name = '%s/%s/%s' % (options.env, options.sub_exp_name, options.exp_name)
    else:
        exp_name = '%s/%s/%s/%s' % (options.env, options.noise, options.sub_exp_name, options.exp_name)

    # load experimental params from json file
    params = process_params(options, options.param_path)

    main(params, exp_name)
