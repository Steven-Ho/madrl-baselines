import numpy as np 
import random
import torch
from functools import reduce
from torch.optim import Adam
import torch.nn.functional as F 
from utils import soft_update, hard_update
from QMIX.model import QMIXNetwork, RNNQNetwork, RNNGaussianPolicy
from utils import soft_update, hard_update
from QMIX.buffer import ReplayMemory

class AgentsTrainer(object):
    def __init__(self, num_agents, obs_shape_list, action_shape_list, args):
        self.na = num_agents
        self.args = args

        self.target_update_interval = args.target_update_interval
        self.tau = args.tau
        self.alpha = 0.
        self.gamma = args.gamma
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # Suppose agents are homogeneous
        # Use critics and actors with shared parameters
        # Therefore agent id (onehot vector) is needed
        self.critics = RNNQNetwork(num_agents + obs_shape_list[0], action_shape_list[0], args.hidden_dim).to(device=self.device)
        self.critics_target = RNNQNetwork(num_agents + obs_shape_list[0], action_shape_list[0], args.hidden_dim).to(device=self.device)
        self.critics_optim = Adam(self.critics.parameters(), lr=args.critic_lr)
        hard_update(self.critics_target, self.critics)

        self.actors = RNNGaussianPolicy(num_agents + obs_shape_list[0], action_shape_list[0], args.hidden_dim).to(device=self.device)
        self.actors_target = RNNGaussianPolicy(num_agents + obs_shape_list[0], action_shape_list[0], args.hidden_dim).to(device=self.device)
        self.actors_optim = Adam(self.actors.parameters(), lr=args.policy_lr)
        hard_update(self.actors_target, self.actors)

        self.qmix_net = QMIXNetwork(num_agents, args.hidden_dim, sum(obs_shape_list)).to(device=self.device)
        self.qmix_net_target = QMIXNetwork(num_agents, args.hidden_dim, sum(obs_shape_list)).to(device=self.device)
        self.qmix_net_optim = Adam(self.qmix_net, lr=args.critic_lr)
        hard_update(self.qmix_net_target, self.qmix_net)

    def act(self, obs_list, eval=False):
        actions = []
        for i in range(self.na):
            obs = torch.FloatTensor(obs_list[i]).to(device=self.device)
            if eval:
                _, _, action = self.actors_target[i].sample(obs)
            else:
                action, _, _ = self.actors_target[i].sample(obs)
            actions.append(action)

        return actions
    
    def make_input(self, obs_list):
        

    def reset(self):
        for i in range(self.na):
            self.actors_target[i].reset()
            self.actors[i].reset()
            self.critics_target[i].reset()
            self.critics[i].reset()

    def update_parameters(self, samples, batch_size, updates):
