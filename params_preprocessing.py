import os
import json


def process_params(options, param_path=None):
    if options.env in [
            'half_cheetah', 'ant', 'walker2d',
            'hopper', 'cheetah_run']:
        if param_path is None:
            param_path = os.path.join(os.path.curdir, 'configs/params_%s.json' % options.env)
    else:
        raise NotImplementedError

    if options.algo not in ['trpo']:
        raise NotImplementedError

    with open(param_path, 'r') as f:
        params = json.load(f)
    params['algo'] = options.algo
    params['noise'] = options.noise
    params['alpha'] = options.alpha  # hyperparam for scaling KL
    params['target_kl'] = options.target_kl  # step size of TRPO
    params['onpol_iters'] = options.onpol_iters
    params['interval'] = options.interval
    params['max_path_length'] = options.max_path_length  # horizon of rollout
    params['trpo_batch_size'] = options.trpo_batch_size
    params['random_seeds'] = options.random_seeds
    params['n_train'] = options.n_train
    params['param_value'] = options.param_value
    params['save_variables'] = options.save_variables
    params['restart_iter'] = options.restart_iter
    params['restore_bc_variables'] = options.restore_bc_variables
    params['restore_policy_variables'] = options.restore_policy_variables
    params['restore_dynamics_variables'] = options.restore_dynamics_variables
    params['restore_offline_data'] = options.restore_offline_data
    params['restore_value'] = options.restore_value
    params['ensemble_model_count'] = options.ensemble_model_count
    params['bc_init'] = options.bc_init
    params["use_s_t"] = options.use_s_t
    params["use_s_0"] = options.use_s_0
    params["pretrain_value"] = options.pretrain_value
    params["video_dir"] = options.video_dir
    params["restore_path"] = options.restore_path
    params['gaussian'] = options.gaussian
    params['const_sampling'] = options.const_sampling
    params['all_bc'] = options.all_bc
    params['eval_model'] = options.eval_model
    params['exp_dir'] = None

    assert params['env_name'] == options.env
    assert params['noise'] in ['pure', 'eps1', 'eps3', 'gaussian1', 'gaussian3', 'random']
    # data_file
    if params['env_name'] == 'ant':
        params['data_file'] = "./data/Ant/{}/data".format(params['noise'])
    elif params['env_name'] == 'half_cheetah':
        params['data_file'] = "./data/HalfCheetah/{}/data".format(params['noise'])
    elif params['env_name'] == 'hopper':
        params['data_file'] = "./data/Hopper/{}/data".format(params['noise'])
    elif params['env_name'] == 'walker2d':
        params['data_file'] = "./data/Walker2d/{}/data".format(params['noise'])

    return params
