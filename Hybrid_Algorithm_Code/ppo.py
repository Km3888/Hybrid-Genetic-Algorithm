import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces
import sys
import numpy as np
import argparse
import os
from torch.distributions import Categorical
from torch.distributions import Normal
import time
from torch import Tensor
from mlp import Net_Continuous, Net_Baseline
from common import get_reward_to_go, get_gae_advantage, get_trajectories, get_tensors_from_paths

## contains the ppo update code

def ppo_update(policy_model, policy_optimizer, baseline_model, baseline_optimizer, baseline_criterion,
               ppo_epoch, minibatch_size, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n,
               gamma, lam, update_policy=True, update_baseline=True):
    """
    Simply throw all the required stuff in this function, and your model will be updated.
    :param policy_model:
    :param policy_optimizer:
    :param baseline_model:
    :param baseline_optimizer:
    :param baseline_criterion:
    :param ppo_epoch: number of ppo epoches to run
    :param minibatch_size: size of a ppo minibatch
    :param obs_n_tensor:
    :param log_prob_n_old_tensor: should be in sum of log form
    :param action_n_tensor:
    :param rewards_n:
    :param mask_n:
    :param gamma:
    :param lam:
    :param update_policy: whether or not to update the policy
    :param update_baseline: whether or not to update the baseline
    """

    ## get number of data
    n_data = obs_n_tensor.shape[0]
    ## get baseline estimations
    baseline_n = baseline_model(obs_n_tensor).detach()  ## between -1 and 1
    ## get q values, used for scaling baseline values
    q_n = get_reward_to_go(rewards_n, mask_n, gamma)
    q_n = Tensor(q_n).reshape(-1, 1)
    q_n_mean = q_n.mean()
    q_n_std = q_n.std() + 1e-2
    ## get scaled baseline values
    baseline_n_scaled = baseline_n * q_n_std + q_n_mean

    ## get advantage and scaled baseline targets
    ## the adv returned is normalized, baseline_target_scaled is not normalized
    adv_n, baseline_target_n_scaled = get_gae_advantage(rewards_n, baseline_n_scaled, mask_n, gamma, lam)
    adv_n = adv_n.detach()

    ## get baseline targets
    baseline_target_n = ((baseline_target_n_scaled - q_n_mean) / q_n_std).detach()  ## target now is normalized

    ## NOTE MAKE SURE YOU DETACH THINGS OTHERWISE YOU WILL HAVE VERY STRANGE ERRORS

    ## for each training iteration, do some epoches
    for i_epoch in range(ppo_epoch):
        ## ppo: shuffle data
        shuffle_indexes = torch.randperm(n_data)
        obs_n_tensor = obs_n_tensor[shuffle_indexes]
        adv_n = adv_n[shuffle_indexes]
        log_prob_n_old_tensor = log_prob_n_old_tensor[shuffle_indexes]
        action_n_tensor = action_n_tensor[shuffle_indexes]
        baseline_target_n = baseline_target_n[shuffle_indexes]

        ## after shuffle data, we do the minibatch ppo update
        n_minibatch = int(n_data / minibatch_size)
        for i_minibatch in range(n_minibatch):
            ## get minibatch
            istart = i_minibatch * minibatch_size
            iend = (i_minibatch + 1) * minibatch_size
            obs_m = obs_n_tensor[istart:iend]
            adv_m = adv_n[istart:iend]
            log_prob_old_m = log_prob_n_old_tensor[istart:iend]
            action_m = action_n_tensor[istart:iend]
            baseline_target_m = baseline_target_n[istart:iend]

            ## update baseline
            if update_baseline:
                baseline_m = baseline_model(obs_m)
                baseline_optimizer.zero_grad()
                baseline_loss = baseline_criterion(baseline_m, baseline_target_m)
                baseline_loss.backward()
                baseline_optimizer.step()

            ## update policy
            ## we need the new policy's log probs so that we can calculate importance sampling term
            if update_policy:
                mu, log_sigma = policy_model(obs_m)
                normal_dist = Normal(mu, log_sigma.exp())
                log_prob_new_m = normal_dist.log_prob(action_m)
                log_prob_new_m = torch.sum(log_prob_new_m, dim=1).reshape(-1, 1)
                ## now we get the importance sampling weight term
                is_term_m = (log_prob_new_m - log_prob_old_m).exp()

                ## compute the clipped surrogate
                epsilon = 0.2
                first_term = is_term_m * adv_m
                clipped_term = is_term_m.clamp(1 - epsilon, 1 + epsilon) * adv_m
                obj_term = torch.min(first_term, clipped_term)
                obj_sum = obj_term.sum()
                policy_loss = -obj_sum / n_data
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()


