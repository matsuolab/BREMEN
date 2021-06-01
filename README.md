# Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization
This codebase implements a deployment-efficient algorithm, BREMEN, proposed in [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/abs/2006.03647).

We modified [ME-TRPO](https://github.com/WilsonWangTHU/mbbl-metrpo) repository for deployment-efficient or offline settings.


## Dependencies
We recommend you to use Docker.

You can use Python 3.6.
You must download MuJoCo 1.31 from https://www.roboti.us/, and then install package dependencies.

```
pip install -r requirements.txt
```


## (Option) Collect data with another codebase
You must use [Behavior Regularized Offline Reinforcement Learning](https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl) codebase for data collection.
Follow their instruction and collect 1M transitions with each noise strategies (pure, eps1, eps3, gaussian1, gaussian3).
If you are interested in deployment-efficient settings, it is enough to collect transitions with *pure* strategy.
After the data collection, put `data.data-00000-of-00001` and `data.index` to the `./data/<Agent name>/pure/`

e.g. `./data/Ant/pure/data.data-00000-of-00001`, `./data/Ant/pure/data.index`

*Note*: This procedure is needed for offline experiments.
If you just run deployment-efficient experiments, you can skip.
However, this must be done if you want to *save video* of your policy (because of the normalization of state and action).

## (Option) Visualize deployment-efficient RL results
This repository contains pre-trained policies of BREMEN in deployment-efficient settings with batch size 200k (Top row in Figure 2).
Save video for the visualization of the results using the following command:

e.g.
```
python save_video.py --env ant --param_path configs/params_ant_offline.json --video_dir <relative path to the video save dir> --restore_path ./weights/Ant/policy.ckpt --restore_policy_variables --n_train 50000
```

You can use four pre-trained policies of BREMEN `ant`, `half_cheetah`, `hopper`, `walker2d`.
(This process requires offline data for the normalization of state and action.)

## Run deployment-efficient experiments
Run BREMEN in deployment-efficient experiments using the following command:

```
python recursive.py --env <env_name> --exp_name <experiment_name> --sub_exp_name <exp_save_dir> --param_path configs/params_<env_name>_offline.json --bc_init --random_seeds 0 --target_kl 0.01 --max_path_length 1000
```

- `env_name`: `ant`, `half_cheetah`, `hopper`, `walker2d`, `cheetah_run`
- `exp_name`: what you want to call your experiment
- `sub_exp_name`: partial path for saving experiment logs and results
- `param_path`: path to config json file
- `target_kl`: delta in TRPO objective
- `max_path_length`: length of an imaginary rollout
- `bc_init`: enable behavior-initialization
- `alpha`: coefficient of explicit KL value penalty (0 is the default)

Experiment results will be logged to `./log/<env_name>/<exp_save_dir>/<experiment_name>/<experiment_name><seed>/`

e.g.
```
python recursive.py --env ant --exp_name recursive_example --sub_exp_name BREMEN_demo --param_path configs/params_ant_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 250 --gaussian 0.1 --const_sampling

python recursive.py --env half_cheetah --exp_name recursive_example --sub_exp_name BREMEN_demo --param_path configs/params_half_cheetah_offline.json --bc_init --random_seeds 0 --target_kl 0.1 --max_path_length 250 --gaussian 0.1 --const_sampling

python recursive.py --env cheetah_run --exp_name recursive_example --sub_exp_name BREMEN_demo --param_path configs/params_cheetah_run_offline.json --bc_init --random_seeds 0 --target_kl 0.1 --max_path_length 250 --gaussian 0.1 --const_sampling

python recursive.py --env hopper --exp_name recursive_example --sub_exp_name BREMEN_demo --param_path configs/params_hopper_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 1000 --gaussian 0.1 --const_sampling --n_train 2000000 --onpol_iters 2400 --interval 240

python recursive.py --env walker2d --exp_name recursive_example --sub_exp_name BREMEN_demo --param_path configs/params_walker2d_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 1000 --gaussian 0.1 --const_sampling --n_train 2000000 --onpol_iters 800
```


## Run offline experiments
Run BREMEN in offline experiments using the following command:

```
python offline.py --env <env_name> --exp_name <experiment_name> --sub_exp_name <exp_save_dir> --param_path configs/params_<env_name>_offline.json --bc_init --random_seeds 0 --target_kl 0.01 --max_path_length 1000
```

- `env_name`: `ant`, `half_cheetah`, `hopper`, `walker2d`
- `exp_name`: what you want to call your experiment
- `sub_exp_name`: partial path for saving experiment logs and results
- `param_path`: path to config json file
- `target_kl`: delta in TRPO objective
- `max_path_length`: length of an imaginary rollout
- `bc_init`: enable behavior-initialization
- `alpha`: coefficient of explicit KL value penalty (0 is the default)
- `onpol_iters`: number of outer iteration (inner iteration is set to 25).
- `noise`: `(pure, eps1, eps3, gaussian1, gaussian3, random)`, default is `pure`

Experiment results will be logged to `./log/<env_name>/<exp_save_dir>/<experiment_name>/<experiment_name><seed>/`

e.g.
```
python offline.py --env ant --exp_name offline_example --sub_exp_name BREMEN_demo --param_path configs/params_ant_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 250 --gaussian 0.1 --const_sampling --onpol_iters 250

python offline.py --env half_cheetah --exp_name offline_example --sub_exp_name BREMEN_demo --param_path configs/params_half_cheetah_offline.json --bc_init --random_seeds 0 --target_kl 0.1 --max_path_length 250 --gaussian 0.1 --const_sampling  --onpol_iters 250

python offline.py --env cheetah_run --exp_name offline_example --sub_exp_name BREMEN_demo --param_path configs/params_cheetah_run_offline.json --bc_init --random_seeds 0 --target_kl 0.1 --max_path_length 250 --gaussian 0.1 --const_sampling  --onpol_iters 250

python offline.py --env hopper --exp_name offline_example --sub_exp_name BREMEN_demo --param_path configs/params_hopper_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 1000 --gaussian 0.1 --const_sampling --onpol_iters 250

python offline.py --env walker2d --exp_name offline_example --sub_exp_name BREMEN_demo --param_path configs/params_walker2d_offline.json --bc_init --random_seeds 0 --target_kl 0.05 --max_path_length 1000 --gaussian 0.1 --const_sampling  --onpol_iters 250
```

## Citation
Please use the following bibtex for citations:
```
@inproceedings{matsushima2020deploy,
    title={Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization},
    author={Tatsuya Matsushima and Hiroki Furuta and Yutaka Matsuo and Ofir Nachum and Shixiang Shane Gu},
    year={2021},
    booktitle={International Conference on Learning Representations},
}
```
