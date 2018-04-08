"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import gym
import numpy as np
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.diayn import DIAYN
from rlkit.torch.networks import FlattenMlp
# from rlkit.torch.networks import Discriminator


def experiment(variant):
    env = NormalizedBoxEnv(gym.make('HalfCheetah-v2'))

    num_skills = variant['num_skills']
    '''observation dim includes dim of latent variable'''
    obs_dim = int(np.prod(env.observation_space.shape)) + num_skills
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )

    # TODO: VERIFY THIS 
    # num_skills=variant['num_skills']
    discrim = FlattenMlp(
        hidden_sizes=[net_size,net_size],
        input_size=obs_dim-num_skills,
        output_size=num_skills,
        output_activation=nn.Sigmoid()
        )

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = DIAYN(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        discrim=discrim,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,
            
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        num_skills=5,
        net_size=300,
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
