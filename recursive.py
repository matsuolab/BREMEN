import argparse
import itertools
import numpy as np
import tensorflow as tf
import time
from libs.misc.initial_configs.algo_config import create_trpo_algo
from libs.misc.initial_configs.policy_config import create_policy_from_params, create_controller_from_policy, \
    create_behavior_policy_from_params

import logger
from libs.misc.data_handling.utils import add_path_data_to_collection_and_update_normalization, \
    replace_path_data_to_collection_and_update_normalization
from libs.misc.data_handling.data_collection import DataCollection
from libs.misc.data_handling.path_collection import PathCollection
from libs.misc.data_handling.rollout_sampler import RolloutSampler
from libs.misc.initial_configs.dynamics_model_config import create_dynamics_model
from libs.misc.saving_and_loading import save_cur_iter_dynamics_model, \
    save_cur_iter_behavior_policy, save_cur_iter_policy, save_cur_iter_offline_data, \
    restore_policy, restore_model, restore_behavior_policy, restore_behavior_policy, \
    restore_offline_data, confirm_restoring_policy, confirm_restoring_dynamics_model, \
    confirm_restoring_behavior_policy, confirm_restoring_offline_data, confirm_restoring_value
from libs.misc.utils import get_session, get_env, get_inner_env
from params_preprocessing import process_params


def log_tabular_results(returns, itr, train_collection):
    logger.clear_tabular()
    logger.record_tabular('Iteration', itr)
    logger.record_tabular('episode_mean', np.mean(returns))
    logger.record_tabular('episode_min', np.min(returns))
    logger.record_tabular('episode_max', np.max(returns))
    logger.record_tabular('TotalSamples', train_collection.get_total_samples())

    logger.dump_tabular()


def get_data_from_random_rollouts(params, env, random_paths, normalization_scope=None, model='dynamics', split_ratio=0.666667):
    train_collection = DataCollection(
        batch_size=params[model]['batch_size'],
        max_size=params['max_train_data'], shuffle=True
    )
    val_collection = DataCollection(
        batch_size=params[model]['batch_size'],
        max_size=params['max_val_data'], shuffle=False
    )
    path_collection = PathCollection()
    obs_dim = env.observation_space.shape[0]
    normalization = add_path_data_to_collection_and_update_normalization(
        random_paths, path_collection, train_collection, val_collection,
        normalization=None,
        split_ratio=split_ratio,
        obs_dim=obs_dim,
        normalization_scope=normalization_scope
    )
    return train_collection, val_collection, normalization, path_collection


def get_data_from_offline_batch(params, env, normalization_scope=None, model='dynamics', split_ratio=0.666667):
    train_collection = DataCollection(
        batch_size=params[model]['batch_size'],
        max_size=params['max_train_data'], shuffle=True
    )
    val_collection = DataCollection(
        batch_size=params[model]['batch_size'],
        max_size=params['max_val_data'], shuffle=False
    )
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


def train_policy_trpo(params, algo, dyn_model, iterations, sess):
    algo.start_worker()
    for j in range(iterations):
        paths, _ = algo.obtain_samples(j, dynamics=dyn_model)
        start = time.time()
        samples_data = algo.process_samples(j, paths)
        end_value_eval = time.time()
        print("value evaluating time: {}".format(end_value_eval - start) + "[sec]")
        algo.optimize_policy(j, samples_data)
        end_policy_update = time.time()
        print("policy optimization time: {}".format(end_policy_update - end_value_eval) + "[sec]")
        algo.fit_baseline(paths)
        end_value_fit = time.time()
        print("value fitting time: {}".format(end_value_fit - end_policy_update) + "[sec]")

    algo.shutdown_worker()


def train(params):
    sess = get_session(interactive=True)
    env = get_env(params['env_name'], params.get('video_dir'))
    inner_env = get_inner_env(env)

    num_paths = int(params['n_train']*params['interval']/params['onpol_iters']/params['env_horizon'])

    rollout_sampler = RolloutSampler(env)
    behavior_policy_rollout_sampler = RolloutSampler(env)
    random_paths = rollout_sampler.generate_random_rollouts(
        num_paths=num_paths,
        horizon=params['env_horizon']
    )

    # get random traj
    train_collection, val_collection, normalization, path_collection = \
        get_data_from_random_rollouts(params, env, random_paths, split_ratio=0.85)

    behavior_policy_train_collection, behavior_policy_val_collection, \
        behavior_policy_normalization, behavior_policy_path_collection = \
        get_data_from_random_rollouts(params, env, random_paths, normalization_scope='behavior_policy', model='behavior_policy', split_ratio=1.0)

    # ############################################################
    # ############### create computational graph #################
    # ############################################################
    policy = create_policy_from_params(params, env, sess)
    controller = create_controller_from_policy(policy)
    rollout_sampler.update_controller(controller)

    # (approximated) behavior policy
    behavior_policy = create_behavior_policy_from_params(params, env, sess)
    behavior_policy_controller = create_controller_from_policy(behavior_policy)
    behavior_policy_rollout_sampler.update_controller(behavior_policy_controller)

    dyn_model = create_dynamics_model(params, env, normalization, sess)

    if params['algo'] not in ('trpo', 'vime'):
        raise NotImplementedError

    algo = create_trpo_algo(
        params, env, inner_env,
        policy, dyn_model, sess,
        behavior_policy=behavior_policy,
        offline_dataset=behavior_policy_train_collection.data["observations"])

    # ############################################################
    # ######################### learning #########################
    # ############################################################

    # init global variables
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)
    policy_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
    behavior_policy_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="behavior_policy")
    if params['param_value']:
        value_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="baseline")
        all_var_except_policy = [v for v in all_variables if v not in (policy_variables + behavior_policy_variables + value_variables)]
    else:
        all_var_except_policy = [v for v in all_variables if v not in (policy_variables + behavior_policy_variables)]

    train_dyn_with_intrinsic_reward_only = params["dynamics"].get("intrinsic_reward_only", False)
    logger.log("Train dynamics model with intrinsic reward only? {}".format(train_dyn_with_intrinsic_reward_only))

    dynamics_saver = tf.train.Saver(var_list=all_var_except_policy)
    behavior_policy_saver = tf.train.Saver(var_list=behavior_policy_variables)
    policy_saver = tf.train.Saver(var_list=policy_variables)
    tf.global_variables_initializer().run()

    if params['restart_iter'] != 0:
        start_itr = params['restart_iter'] + 1
    else:
        start_itr = params.get("start_onpol_iter", 0)
    interval = params['interval']
    end_itr = params['onpol_iters']

    if train_dyn_with_intrinsic_reward_only:
        # Note: not supported
        dyn_model.use_intrinsic_rewards_only()
    else:
        dyn_model.use_external_rewards_only()

    # for restart experiment
    if confirm_restoring_policy(params):
        restore_policy(params, policy_saver, sess)
    if confirm_restoring_dynamics_model(params):
        restore_model(params, dynamics_saver, sess)
    if confirm_restoring_behavior_policy(params):
        restore_behavior_policy(params, behavior_policy_saver, sess)
    if confirm_restoring_offline_data(params):
        train_collection, val_collection, behavior_policy_train_collection = restore_offline_data(params)
        policy.running_stats.update_stats(train_collection.data["observations"])
        behavior_policy.running_stats.update_stats(
            behavior_policy_train_collection.data["observations"]
        )
    if confirm_restoring_value(params):
        algo.baseline.restore_value_function(params['restore_path'], params['restart_iter'])
        algo.baseline.running_stats.update_stats(train_collection.data["observations"])

    # training
    for itr in range(start_itr, end_itr):
        if itr % interval == 0:
            if itr != 0:
                logger.info("Collecting offline data with online interaction.")
                rl_paths = rollout_sampler.sample(
                    num_paths=num_paths,
                    horizon=params['env_horizon'],
                    evaluation=False
                )
                # Update data for dynamics training
                normalization = add_path_data_to_collection_and_update_normalization(
                    rl_paths, path_collection, train_collection,
                    val_collection, normalization, split_ratio=0.85
                )
                # Update data for BC fitting
                if not params['all_bc']:
                    behavior_policy_normalization = replace_path_data_to_collection_and_update_normalization(
                        rl_paths, behavior_policy_train_collection,
                        behavior_policy_val_collection,
                        behavior_policy_normalization, split_ratio=1.0
                    )
                else:
                    behavior_policy_normalization = add_path_data_to_collection_and_update_normalization(
                        rl_paths, behavior_policy_path_collection,
                        behavior_policy_train_collection,
                        behavior_policy_val_collection,
                        behavior_policy_normalization, split_ratio=1.0
                    )
                behavior_policy_train_collection.set_batch_size(params['behavior_policy']['batch_size'])
            # dynamics
            logger.info("Fitting dynamics.")
            dyn_model.fit(train_collection, val_collection)
            logger.info("Done fitting dynamics.")
            save_cur_iter_dynamics_model(
                params, dynamics_saver, sess, itr
            )
            rollout_sampler.update_dynamics(dyn_model)
            # BC
            logger.info("Fitting BC.")
            behavior_policy.initialize_variables()
            behavior_policy.running_stats.update_stats(
                behavior_policy_train_collection.data["observations"]
            )
            behavior_policy.fit_as_bc(
                behavior_policy_train_collection,
                behavior_policy_val_collection,
                behavior_policy_rollout_sampler
            )
            save_cur_iter_behavior_policy(
                params, behavior_policy_saver, sess, itr
            )
            logger.info("Done fitting BC.")
            # re-initialize TRPO policy with BC policy
            if params['bc_init']:
                logger.info("Initialize TRPO policy with BC.")
                update_weights = [
                    tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('policy'), tf.trainable_variables('behavior_policy'))
                ]
                sess.run(update_weights)
                algo.reinit_with_source_policy(behavior_policy)
                if rollout_sampler:
                    rl_paths = rollout_sampler.sample(
                        num_paths=params['num_path_onpol'],
                        horizon=params['env_horizon'],
                        evaluation=True
                    )
                returns = np.mean(np.array([sum(path["rewards"]) for path in rl_paths]))
                logger.info("TRPO policy initialized with BC average return: {}".format(returns))

            if params['pretrain_value']:
                logger.info("Fitting value function.")
                behavior_policy_train_collection.set_batch_size(params['max_path_length'])
                for obses, _, _, rewards in behavior_policy_train_collection:
                    algo.pre_train_baseline(obses, rewards, params['trpo']['gamma'], params['trpo']['gae'])
                logger.info("Done fitting value function.")

            save_cur_iter_offline_data(
                params, train_collection, val_collection, behavior_policy_train_collection, itr,
            )

        logger.info('itr #%d | ' % itr)

        # Update randomness
        logger.info("Updating randomness.")
        dyn_model.update_randomness()
        logger.info("Done updating randomness.")

        # Policy training
        logger.info("Training policy using TRPO.")
        train_policy_trpo(params, algo, dyn_model, params['trpo']['iterations'], sess)
        logger.info("Done training policy.")

        # Generate on-policy rollouts.
        # only for evaluation, not for updating data
        logger.info("Generating on-policy rollouts.")
        if params['eval_model']:
            rl_paths, rollouts, residuals = rollout_sampler.sample(
                num_paths=params['num_path_onpol'],
                horizon=params['env_horizon'],
                evaluation=True,
                eval_model=params['eval_model']
            )
        else:
            rl_paths = rollout_sampler.sample(
                num_paths=params['num_path_onpol'],
                horizon=params['env_horizon'],
                evaluation=True
            )
        logger.info("Done generating on-policy rollouts.")
        returns = np.array([sum(path["rewards"]) for path in rl_paths])
        log_tabular_results(returns, itr, train_collection)
        if params['eval_model']:
            n_transitions = sum([len(path["rewards"]) for path in rl_paths])
            # step_wise_analysis
            step_wise_mse = np.mean([sum(np.array(path["observations"])**2) for path in residuals])
            step_wise_mse /= n_transitions
            logger.record_tabular('step_wise_mse', step_wise_mse)
            step_wise_episode_mean = np.mean([sum(path["rewards"]) for path in residuals])
            logger.record_tabular('step_wise_episode_mean', step_wise_episode_mean)
            # trajectory_wise_analysis
            min_path = min([len(path["observations"]) for path in rl_paths])
            min_rollout = min([len(rollout["observations"]) for rollout in rollouts])
            traj_len = min(min_path, min_rollout)
            traj_wise_mse = np.mean([sum((np.array(path["observations"])[:traj_len]-np.array(rollout["observations"])[:traj_len])**2) for (path, rollout) in zip(rl_paths, rollouts)])
            traj_wise_mse /= traj_len*params['num_path_onpol']
            logger.record_tabular('traj_wise_mse', traj_wise_mse)
            traj_wise_episode_mean = np.mean([sum(path["rewards"][:traj_len]) for path in rollouts])
            logger.record_tabular('traj_wise_episode_mean', traj_wise_episode_mean)
            # Energy distance between \tau_{sim} and \tau_{real}
            combination_sim_real = list(itertools.product(rl_paths, rollouts))
            A = np.mean([sum(np.sqrt((np.array(v[0]["observations"][:traj_len]) - np.array(v[1]["observations"][:traj_len]))**2)) for v in combination_sim_real])
            combination_sim = list(itertools.product(rollouts, rollouts))
            B = np.mean([sum(np.sqrt((np.array(v[0]["observations"][:traj_len]) - np.array(v[1]["observations"][:traj_len]))**2)) for v in combination_sim])
            combination_real = list(itertools.product(rl_paths, rl_paths))
            C = np.mean([sum(np.sqrt((np.array(v[0]["observations"][:traj_len]) - np.array(v[1]["observations"][:traj_len]))**2)) for v in combination_real])
            energy_dist = np.sqrt(2*A-B-C)
            logger.record_tabular('energy_distance', energy_dist)
            logger.dump_tabular()
        if itr % interval == 0 or itr == end_itr-1:
            save_cur_iter_policy(params, policy_saver, sess, itr)
            if params['save_variables']:
                algo.baseline.save_value_function(params['exp_dir'], itr)


def get_exp_name(root_dir, exp_name, seed, noise):
    if noise == 'pure':
        return root_dir + exp_name + '_seed' + str(seed)
    else:
        return root_dir + exp_name + '_seed' + str(seed) + noise


def set_seed(seed):
    seed %= 4294967294
    global seed_
    seed_ = seed
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except Exception as e:
        print(e)


def run_train(params, exp_name):
    for seed in params["random_seeds"]:
        # set seed
        print("Using random seed {}".format(seed))
        set_seed(seed)

        # logger
        exp_dir = get_exp_name(params["log_save_dir"], exp_name, seed, params['noise'])
        params['exp_dir'] = exp_dir
        logger.configure(exp_dir)
        logger.info("Print configuration .....")
        logger.info(params)
        train(params)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiment options')
    parser.add_argument('--env')
    parser.add_argument('--exp_name')
    parser.add_argument('--sub_exp_name', default="")
    parser.add_argument('--noise', default='pure')
    parser.add_argument('--algo', default='trpo')
    parser.add_argument('--param_path', default=None)
    parser.add_argument('--onpol_iters', type=int, default=400)
    parser.add_argument('--interval', type=int, default=80)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--trpo_batch_size', type=int, default=50000)
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[1234, 4321, 2341, 3341, 789])
    parser.add_argument('--n_train', type=int, default=1000000)
    parser.add_argument('--alpha', type=float, default=0.)  # hyperparam for scaling KL
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

    exp_name += '/%s' % (options.exp_name)

    # load experimental params from json file
    params = process_params(options, options.param_path)

    run_train(params, exp_name)
