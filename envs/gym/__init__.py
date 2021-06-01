from gym.envs.registration import register


register(
    id='MBRLHalfCheetah-v0',
    entry_point='envs.gym.half_cheetah:HalfCheetahEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLCheetahRun-v0',
    entry_point='envs.gym.cheetahrun:CheetahRunEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLWalker2d-v0',
    entry_point='envs.gym.walker2d:Walker2dEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLAnt-v0',
    entry_point='envs.gym.ant:AntEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLHopper-v0',
    entry_point='envs.gym.hopper:HopperEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLFetchReachDense-v0',
    entry_point='envs.gym.robotics.fetch.reach:FetchReachEnv',
    max_episode_steps=50,
)

register(
    id='MBRLFetchPushDense-v0',
    entry_point='envs.gym.robotics.fetch.push:FetchPushEnv',
    max_episode_steps=50,
)

register(
    id='MBRLHandReachDense-v0',
    entry_point='envs.gym.robotics.hand.reach:HandReachEnv',
    max_episode_steps=50,
)

env_name_to_gym_registry = {
    "half_cheetah": "MBRLHalfCheetah-v0",
    "ant": "MBRLAnt-v0",
    "hopper": "MBRLHopper-v0",
    "walker2d": "MBRLWalker2d-v0",
    "cheetah_run": "MBRLCheetahRun-v0",
    "fetchpush_dense": "MBRLFetchPushDense-v0",
    "fetchreach_dense": "MBRLFetchReachDense-v0",
    "handreach_dense": "MBRLHandReachDense-v0",
}
