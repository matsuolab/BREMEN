from envs.neural_env import NeuralNetEnv
from model.baselines import LinearFeatureBaseline, MLPBaseline
from algos.trpo import TRPO as TRPO_mbrl


def get_base_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=None, behavior_policy=None, offline_dataset=None):
    neural_env = NeuralNetEnv(
        env=env, inner_env=inner_env,
        dynamics=dyn_model,
        offline_dataset=offline_dataset,
        use_s_t=params['use_s_t'], use_s_0=params['use_s_0']
    )

    if params['param_value']:
        baseline = MLPBaseline(name="baseline", observation_space=neural_env.observation_space, sess=sess)
    else:
        baseline = LinearFeatureBaseline(name="baseline")
    return {
        "env": neural_env,
        "inner_env": inner_env,
        "policy": policy,
        "baseline": baseline,
        "batch_size": params['trpo_batch_size'],
        "max_path_length": params['max_path_length'],
        "discount": params['trpo']['gamma'],
        "target_kl": params['target_kl'],
        "gae_lambda": params['trpo']['gae'],
        "sess": sess,
        "scope": scope,
        "behavior_policy": behavior_policy,
        "alpha": params['alpha'],
        "offline_dataset": offline_dataset,
        "use_s_t": params['use_s_t'],
        "use_s_0": params['use_s_0']
    }


def create_trpo_algo(params, env, inner_env, policy, dyn_model, sess, scope=None, behavior_policy=None, offline_dataset=None):
    if params['algo'] == 'trpo':
        return TRPO_mbrl(**get_base_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=scope, behavior_policy=behavior_policy, offline_dataset=offline_dataset))
    else:
        raise ValueError("To create TRPO algo, params['algo'] must be {}".format("'trpo'"))
