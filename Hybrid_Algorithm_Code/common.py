## common utilities for deep reinforcement learning
import numpy as np
import torch
import time
from torch.distributions import Normal
from torch import Tensor

def get_reward_to_go(rewards:list, mask_n, gamma:float):
    """
    Get reward to go for each timestep. Assume the rewards given are within one trajectory.
    If you have multiple trajectories, call this function for each trajectory separately.
    :param rewards: reward at each time step,
    :param gamma: the discount factor gamma
    :return: a list, at index t: the discounted reward-to-go q(s_t,a_t)
    """
    T = len(rewards)
    q = 0
    q_n = []
    for t in range(T-1,-1,-1): ## use a reverse for loop to reduce computation
        q = rewards[t] + gamma*q * mask_n[t]
        q_n.append(q)
    q_n.reverse() ## need to call reverse
    return q_n

def get_gae_advantage(rewards, values, mask_n,  gamma, lam):
    ## i thnk this is gae advantage
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)#Oh interesting/ well coded, using the type as a variable and calling it on
                                            #desired sizes. rewards.size is the number of rewards
    advantages = tensor_type(rewards.size(0), 1)

    value_next_state = 0 ## value estimated from a value network, from t+1 timestep
    adv_next_state = 0 ## advantage at time t+1
    ## final state's mask is 0, so final state's
    for t in reversed(range(rewards.size(0))):
        ## here prev value means value of last iteration in this for loop
        deltas[t] = rewards[t] + gamma * value_next_state * mask_n[t] - values[t]
        advantages[t] = deltas[t] + gamma * lam * adv_next_state * mask_n[t]
        value_next_state = values[t, 0]
        adv_next_state = advantages[t, 0]

    ## values are the value network estimates, same as used in openai baseline implementation
    value_net_target = values + advantages
    ## normalize advantage
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages, value_net_target


def get_trajectories(env, policy_model, n_episode, episode_length):
    """
    ## get some trajectories (data)
    :return: return a list of paths, and the average episode return
    """
    paths = []
    epoch_return = 0
    episode_count = 0
    for i_episode in range(n_episode):
        episode_return = 0
        episode_count += 1
        obs_list = []
        log_prob_list = []
        action_list = []
        reward_list = []
        mask_list = []
        observation = env.reset()

        for t in range(episode_length):
            # env.render()
            obs_list.append(observation)
            obs_tensor = torch.Tensor(observation).view(1,-1)

            mu, log_sigma = policy_model(obs_tensor)

            normal_dist = Normal(mu,log_sigma.exp())

            action = normal_dist.sample()
            action_list.append(action.reshape(-1))

            ## the log prob here is the log prob of taking the set of actions, so we take sum of log
            ## you can also take product of exp of log to get same results
            log_prob = normal_dist.log_prob(action)
            log_prob = torch.sum(log_prob)
            log_prob_list.append(log_prob)

            ## make sure action is within env's specifics
            action = torch.clamp(action, -1, 1).reshape(-1)
            observation, reward, done, info = env.step(action.data.numpy())

            if done:
                mask_list.append(0) ## used in calculating advantage
            else:
                mask_list.append(1)

            episode_return += reward
            reward_list.append(reward)
            if done:
                break
        epoch_return += episode_return
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
        path = {'obs':obs_list,'mask':mask_list, 'log_probs':log_prob_list, 'rewards':reward_list,'actions':action_list,'episode_return':episode_return}
        paths.append(path)
    return paths, epoch_return/n_episode
    #REturns the paths and

def get_tensors_from_paths(paths):
    """
    given trajectories, this function helps you get everything into nxd tensor form, n is number of data point
    you can also easily extract data using your own functions
    :param paths:
    :return:
    """
    obs_n = []
    log_prob_n = []
    action_n = []
    rewards_n = []
    mask_n = []
    for path in paths:
        obs_n += path['obs']
        log_prob_n += path['log_probs']
        action_n += path['actions']
        rewards_n += path['rewards']
        mask_n += path['mask']
    ## convert each type of data into usable form
    obs_n_tensor = Tensor(np.vstack(obs_n))
    n_data = obs_n_tensor.shape[0]
    ## action probability of old policy
    log_prob_n_old_tensor = torch.stack(log_prob_n).reshape(n_data,-1).detach()
    action_n_tensor = torch.stack(action_n).detach()
    rewards_n = Tensor(rewards_n).reshape(n_data,-1)
    mask_n = Tensor(mask_n).reshape(n_data,-1)
    return n_data, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n

#
# r = torch.Tensor([1,1,1,0]).reshape(-1,1)
# vst = torch.Tensor([2,2,2,0]).reshape(-1,1)
# vstn = torch.Tensor([3,3,3,0]).reshape(-1,1)
# gamma = 0.9
# lam = 0.9
#
# results = get_gae_advantage(r,vst,vstn,gamma,lam)
#
# print(results)
#
#
# def estimate_advantages(rewards, values, gamma, tau):
#     ## i thnk this is gae advantage
#     tensor_type = type(rewards)
#     deltas = tensor_type(rewards.size(0), 1)
#     advantages = tensor_type(rewards.size(0), 1)
#
#     prev_value = 0
#     prev_advantage = 0
#     ## final state's mask is 0, so final state's
#     for i in reversed(range(rewards.size(0))):
#         ## here prev value means value of last iteration in this for loop
#         deltas[i] = rewards[i] + gamma * prev_value * 1 - values[i]
#         advantages[i] = deltas[i] + gamma * tau * prev_advantage * 1
#
#         prev_value = values[i, 0]
#         prev_advantage = advantages[i, 0]
#     print('d2',deltas)
#
#     ## values are the value network estimates, same as used in openai baseline implementation
#     returns = values + advantages
#     ## normalize advantage
#     # advantages = (advantages - advantages.mean()) / advantages.std()
#
#     return advantages
#
# results, _ = estimate_advantages(r, vst,gamma,lam)
# print(results)