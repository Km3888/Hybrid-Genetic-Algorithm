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

LOG_FOLDER_PATH = 'logs'
if not os.path.isdir(LOG_FOLDER_PATH):
    os.mkdir(LOG_FOLDER_PATH)
    print("log directory not found, created folder.")

## add project path if using folders
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import get_reward_to_go, get_gae_advantage, get_trajectories, get_tensors_from_paths

from ppo import ppo_update

## PPO implementation

program_start_time = time.time()
parser = argparse.ArgumentParser()

#'InvertedPendulum-v2'
#'HalfCheetah-v2'
parser.add_argument('--envname', type=str, default='HalfCheetah-v2', help='name of the gym env')
# parser.add_argument("--max_timesteps", type=int, default=)
parser.add_argument('-n','--num_train_iter', type=int, default=100,
                    help='Number of training iterations')
parser.add_argument('-ep','--episode_length', type=int, default=100,
                    help='Number of max episode length')
parser.add_argument('-seed','--random_seed', type=int, default=42,
                    help='random seed')
parser.add_argument('-nep', '--n_episode', type=int, default=21,
                    help='number of episode to run for each training iteration')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('-e','--n_experiment', type=int, default=3,
                    help='Number of experiment to run')
parser.add_argument('-hd','--hidden_dim', type=int, default=64,
                    help='number of hidden layer neurons')

parser.add_argument('-mb','--minibatch_size', type=int, default=64,
                    help='number of data in one minibatch')

parser.add_argument('--ppo_epoch', type=int, default=10,
                    help='number of ppo epoch in one iteration')

parser.add_argument('-lr','--learning_rate', type=float, default=3e-4,
                    help='model learning rate')
parser.add_argument('-wd','--weight_decay', type=float, default=1e-3,
                    help='model weight decay rate')
parser.add_argument('-lam','--lam', type=float, default=0.95,
                    help='lambda value in generalized advantage estimator')
parser.add_argument('--exp_name', type=str, default='trial_run.txt', help='specify name of log file')

args = parser.parse_args()

DEBUG = False
if DEBUG:
    args.num_train_iter = 100
    print("DEBUG MODE!!!!!!!!!!!!!!!!!!!!!")

print(args)
sys.stdout.flush()

def train_PG(envname,
             num_train_iter,
             random_seed,
             learning_rate,
             gamma,
             episode_length,
            n_episode, hidden_dim,weight_decay,lam, ppo_epoch, minibatch_size
             ):
    ###### seed
    env = gym.make(envname)
    torch.manual_seed(4564)
    np.random.seed(random_seed)
    env.seed(random_seed)
    training_log_single_experiment = []

    ###### get env info
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    n_observation = env.observation_space.shape[0]
    if is_discrete:
        n_action = env.action_space.n
    else:
        n_action = env.action_space.shape[0]

    ###### define model and optimizer
    policy_model = Net_Continuous(n_observation, hidden_dim, n_action)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    baseline_model = Net_Baseline(n_observation, hidden_dim)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay) ## weight decay only for baseline
    baseline_criterion = nn.MSELoss()

    ###### start training
    for i_training in range(num_train_iter):
        ## get trajectories by doing Monte Carlo run on the environment
        paths, average_episode_return = get_trajectories(env, policy_model, n_episode, episode_length)
        print("epoch", i_training, "ave reward:", average_episode_return, "t:", int((time.time() - program_start_time) / 60))
        sys.stdout.flush()
        training_log_single_experiment.append(average_episode_return)

        ## several Monte Carlo roll outs are finished
        ## now we collected a batch of data, the next is do updates

        ## first extract data from paths
        n_data, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n = get_tensors_from_paths(paths)

        ## do a ppo update
        ppo_update(policy_model, policy_optimizer, baseline_model, baseline_optimizer, baseline_criterion,
                   ppo_epoch, minibatch_size, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n,
                   gamma, lam)

    return training_log_single_experiment

return_log_all = [] #Keeps track of all rewards throughout training process
n_experiment = args.n_experiment#How many training processes to be run
for i in range(n_experiment):
    random_seed = (i+1) * args.random_seed
    return_log = train_PG(envname=args.envname,
                          num_train_iter=args.num_train_iter,
                          random_seed=random_seed,
                          learning_rate=args.learning_rate,
                          gamma = args.gamma,
                          episode_length=args.episode_length,
                          n_episode = args.n_episode,
                          hidden_dim=args.hidden_dim,weight_decay=args.weight_decay,
                          lam=args.lam,ppo_epoch=args.ppo_epoch,minibatch_size=args.minibatch_size
                          )
    return_log_all.append(return_log)

training_log_all_experiment = np.array(return_log_all)
save_path = os.path.join('logs',args.exp_name)
np.savetxt(save_path,training_log_all_experiment)

