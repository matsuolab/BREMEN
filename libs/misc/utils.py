import gym
import numpy as np
import tensorflow as tf

from libs.misc.visualization import get_video_recording_status, \
    turn_off_video_recording, turn_on_video_recording


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def get_inner_env(env):
    return env.wrapped_env


def get_session(interactive=False, mem_frac=0.25, use_gpu=True):
    tf.reset_default_graph()
    if use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            gpu_options=gpu_options)
        if interactive:
            session = tf.InteractiveSession(config=tf_config)
        else:
            session = tf.Session(config=tf_config)
        print("AVAILABLE GPUS: ", get_available_gpus())
        return session
    # IF not using gpu
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    if interactive:
        return tf.InteractiveSession(config=config)
    return tf.Session(config=config)


def minimize_and_clip(optimizer, objective, var_list, clip_val=None):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            if clip_val is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def get_env(env_name, video_dir=None):
    from envs.gym import env_name_to_gym_registry
    from envs.proxy_env import ProxyEnv

    unnormalized_env = gym.make(env_name_to_gym_registry[env_name])
    turn_off_video_recording()

    if video_dir:
        turn_on_video_recording()
        unnormalized_env = gym.wrappers.Monitor(unnormalized_env, video_dir)

    def video_callable(_):
        return get_video_recording_status()

    return ProxyEnv(unnormalized_env)
