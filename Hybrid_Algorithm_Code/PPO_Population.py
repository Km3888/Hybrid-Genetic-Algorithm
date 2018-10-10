from common import get_reward_to_go, get_gae_advantage, get_trajectories, get_tensors_from_paths
from ppo import ppo_update
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.optim import Adam
import random
from operator import attrgetter
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()

class Population:

    def __init__ (self,input_dimension,hidden_layer_size,output_dimension,
                 population_size,survival_rate,sigma,policy_gradient,env,
                 n_trials,pg_lr,ppo_epoch,minibatch_size,gamma,lam,episode_length,weight_decay):

        self.total_data=0
        self.data_amounts=[]

        self.population=None

        #Hyperparameters for NNs in population

        self.input_dimension=input_dimension
        self.hidden_layer_size=hidden_layer_size
        self.output_dimension=output_dimension

        #Hyperparameters for growth of population
        self.population_size=population_size
        self.survival_rate=survival_rate
        self.sigma=sigma
        self.env=env
        self.n_trials=n_trials

        #Experiment controls
        self.policy_gradient=policy_gradient
        self.lr=pg_lr
        #Tracker attributes
        self.i_gen=0
        self.best_scores=[]
        self.elite = None

        ##PPO Controls
        self.ppo_epoch =ppo_epoch
        self.minibatch_size =minibatch_size
        self.gamma =gamma
        self.lam =lam
        self.episode_length=episode_length
        self.weight_decay=weight_decay

        self.init_population()

    def init_population(self):


        for i_member in range(self.population_size):

            self.population = [Member(population=self) for _ in range(self.population_size)]

    def gen_update(self):

        #Create an empty array which we can later sort
        next_gen=[]


        #Run rollouts on every member of the population
        for member in self.population:
            env=gym.make(self.env)
            member.paths, member.fitness = get_trajectories(env, member.policy_model, self.n_trials, self.episode_length)
            for path in member.paths:
                self.total_data+=len(path['obs'])

        #Rank population based on fitness
        self.population.sort(key=attrgetter('fitness'))

        #Record the best score in the population

        #Perform PPO updates for every member of the population
        if self.policy_gradient:

            #Here we do the PPO update


            j=-1
            for _ in range(int(1*self.survival_rate*self.population_size)):
                member=self.population[j]
                n_data, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n = get_tensors_from_paths(
                    member.paths)

                ppo_update(policy_model=member.policy_model, policy_optimizer=member.policy_optimizer,baseline_model=member.baseline_model,
                           baseline_optimizer=member.baseline_optimizer, baseline_criterion=member.baseline_criterion,
                           ppo_epoch=self.ppo_epoch, minibatch_size=self.minibatch_size,obs_n_tensor=obs_n_tensor,log_prob_n_old_tensor=log_prob_n_old_tensor,
                           action_n_tensor=action_n_tensor, rewards_n=rewards_n,mask_n=mask_n, gamma=self.gamma, lam=self.lam)


                j-=1

        self.elite=self.population[-1]
        elite_baby=Member(parent=self.elite,sigma=0)
        self.best_scores.append(self.elite.fitness)
        self.data_amounts.append(self.total_data)
        next_gen.append(elite_baby)


        for i_new_pheno in range(1,self.population_size):

            n_survivors=int(self.survival_rate*self.population_size)
            number=random.randint(-1*n_survivors,-1)
            parent=self.population[number]
            #select a random (surviving) member to be a parent

            new_pheno=Member(parent=parent,sigma=self.sigma)


            next_gen.append(new_pheno)#Fill in the new generation!


            #Let's not worry about fitness approximation for right now


                #I should add in a little comparison thing here as a diagnostic to see if pg_updates actually help

        #Babysitting aides:
        print('-'*12)
        print('i_gen=',self.i_gen)
        print('population:',[member.fitness for member in self.population])
        print('best=',self.elite.fitness)
        self.population=next_gen
        self.i_gen+=1

    def string(self):
        string=self.env
        if self.policy_gradient:
            string+='_HybridPPO'
        else:
            string+='_Vanilla'
        string += '_sigma' + str(self.sigma)
        string+='_lr'+str(self.lr)
        string+='_SR'+str(self.survival_rate)
        string+='_gamma'+str(self.gamma)
        string+='_EL'+str(self.episode_length)
        string+='trials'+str(self.n_trials)
        string+='_batch'+str(self.minibatch_size)
        string+='_lam'+str(self.lam)+ '.txt'
        return string

####These two methods need to be vectorized

class Net_Continuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net_Continuous, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim)-2)
        self.init_weights()
    def forward(self, x):
        out = F.relu(self.input(x))
        mu = torch.tanh(self.mu(out))
        action_log_std = self.action_log_std.expand_as(mu)
        return mu, action_log_std

    def init_weights(self):
        for layer in [self.input,self.mu]:
            layer.weight.data.normal_(0,0.01)

class Net_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net_Baseline, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.input = nn.Linear(input_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,1)
    def forward(self, x):
        out = F.relu(self.input(x))
        out = torch.tanh(self.output(out))
        return out

class Member:

    def __init__(self,parent=None,sigma=None,population=None,input_dimension=None,hidden_layer_size=None,output_dimension=None,lr=None
                ):
        if sigma is not None:
            self.sigma = sigma
        if not parent is None:
            if sigma is None:
                self.sigma = parent.sigma
            self.input_dimension = parent.input_dimension
            self.hidden_layer_size = parent.hidden_layer_size
            self.output_dimension = parent.output_dimension
            self.lr=parent.lr
            self.weight_decay=parent.weight_decay

            self.population=parent.population
            self.parent=parent

            self.policy_model=self.mutate(parent.policy_model,self.sigma)
            self.policy_optimizer=self.get_optimizer(self.policy_model,parent.policy_optimizer)
            self.baseline_model=Net_Baseline(parent.baseline_model.input_dim,parent.baseline_model.hidden_dim)
            self.baseline_model.load_state_dict(parent.baseline_model.state_dict())
            self.baseline_optimizer = self.get_optimizer(self.baseline_model,parent.baseline_optimizer)  ## weight decay only for baseline
            self.baseline_criterion = copy.deepcopy(parent.baseline_criterion)


        elif population is not None:
            if sigma is None:
                self.sigma = population.sigma
            self.input_dimension = population.input_dimension
            self.hidden_layer_size = population.hidden_layer_size
            self.output_dimension = population.output_dimension
            self.lr=population.lr
            self.population=population
            self.weight_decay=population.weight_decay



            self.policy_model = Net_Continuous(self.input_dimension, self.hidden_layer_size, self.output_dimension)
            self.policy_optimizer = Adam(self.policy_model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
            self.baseline_model = Net_Baseline(self.input_dimension,self.hidden_layer_size)
            self.baseline_optimizer = Adam(self.baseline_model.parameters(), lr=self.lr,
                                            weight_decay=.99)  ## weight decay only for baseline
            self.baseline_criterion = nn.MSELoss()
        else:
            raise TypeError('Member needs net dimensions')

        self.fitness=None
        self.paths=None

    def get_optimizer(self,model,opt):
        new_opt=Adam(model.parameters(),weight_decay=self.weight_decay)

        #Weird error here where the old optimizer only has 4 params under 'params
        new_opt.load_state_dict(opt.state_dict())
        return new_opt

    def mutate(self,model,sigma):
        new_model=Net_Continuous(self.input_dimension, self.hidden_layer_size, self.output_dimension)
        d = copy.deepcopy(model.state_dict())
        new_dict={}
        for k in d:
            if not sigma==0:
                new_dict[k] = torch.normal(d[k], self.sigma)
            else:
                new_dict[k] = d[k]
        new_model.load_state_dict(new_dict)
        return new_model


        return baby

if __name__=='__main__':
    #argparser
    if True:  # argparser stuff
        parser.add_argument('-env', '--envname', type=str, default='HalfCheetah-v2', help='name of the gym env')

        parser.add_argument('--hybrid', action='store_true', help='Flag for use of hybrid algorithm')

        parser.add_argument('-n', '--n_gen', type=int, default=75,
                            help='Number of generations to train population')

        parser.add_argument('-pop', '--pop_size', type=int, default=100)

        parser.add_argument('-sr', '--survival_rate', type=float, default=.2)

        parser.add_argument('-ep', '--episode_length', type=int, default=200,
                            help='Number of max episode length')

        parser.add_argument('-sig', '--sigma', type=float, default=0.05,
                            help='standard deviation value for gaussian noise')

        parser.add_argument('-nep', '--n_trials', type=int, default=5,
                            help='number of episodes to run for each training iteration')

        parser.add_argument('-g', '--gamma', type=float, default=0.99,
                            help='discount factor')

        parser.add_argument('-e', '--n_experiment', type=int, default=5,
                            help='Number of experiment to run')

        parser.add_argument('-in', '--input_dimension', type=int, default=17,
                            help='Input dimension of model')

        parser.add_argument('-hd', '--hidden_dim', type=int, default=64,
                            help='number of hidden layer neurons')

        parser.add_argument('-out', '--output_dimension', type=int, default=6,
                            help='output dimension of model')

        parser.add_argument('-mb', '--minibatch_size', type=int, default=64,
                            help='number of data in one minibatch')

        parser.add_argument('--ppo_epoch', type=int, default=10,
                            help='number of ppo epochs in one iteration')

        parser.add_argument('-lr', '--lr', type=float, default=0.00001,
                            help='model learning rate')

        parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3,
                            help='model weight decay rate')

        parser.add_argument('-lam', '--lam', type=float, default=0.95,
                            help='lambda value in generalized advantage estimator')

        args = parser.parse_args()
    records=[]
    for i in range(args.n_experiment):
        torch.manual_seed((i+1)**2)
        test_pop=Population(input_dimension=args.input_dimension,hidden_layer_size=args.hidden_dim,
                            output_dimension=args.output_dimension,population_size=args.pop_size,
                            survival_rate=args.survival_rate,sigma=args.sigma,policy_gradient=args.hybrid,
                            env=args.envname,n_trials=args.n_trials,pg_lr=args.lr,
                            ppo_epoch=args.ppo_epoch,minibatch_size=args.minibatch_size,
                            gamma=args.gamma,lam=args.lam,episode_length=args.episode_length,weight_decay=args.weight_decay)
        if i==0:
            name = test_pop.string()

        for _ in range(args.n_gen):
            test_pop.gen_update()
            print(test_pop.best_scores)
            sys.stdout.flush()

        records.append(test_pop.best_scores)


    best_scores_array=np.stack(records)
    np.savetxt(name,best_scores_array)




