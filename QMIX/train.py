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

        self.critics = []
        self.critics_target = []
        self.critic_optims = []
        for i in range(num_agents):
            self.critics.append(RNNQNetwork(obs_shape_list[i], action_shape_list[i], args.hidden_dim).to(device=self.device))
            self.critics_target.append(RNNQNetwork(obs_shape_list[i], action_shape_list[i], args.hidden_dim).to(device=self.device))
            self.critic_optims.append(Adam(self.critics[i].parameters(), lr=args.critic_lr))
            hard_update(self.critics_target[i], self.critics[i])

        self.actors = []
        self.actors_target = []
        self.actor_optims = []
        for i in range(num_agents):
            self.actors.append(RNNGaussianPolicy(obs_shape_list[i], action_shape_list[i], args.hidden_dim).to(device=self.device))
            self.actors_target.append(RNNGaussianPolicy(obs_shape_list[i], action_shape_list[i], args.hidden_dim).to(device=self.device))
            self.actor_optims.append(Adam(self.actors[i].parameters(), lr=args.policy_lr))
            hard_update(self.actors_target[i], self.actors[i])

        self.qmix_net = QMIXNetwork(num_agents, args.hidden_dim, sum(obs_shape_list)).to(device=self.device)

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
    
    def reset(self):
        for i in range(self.na):
            self.actors_target[i].reset()
            self.actors[i].reset()
            self.critics_target[i].reset()
            self.critics[i].reset()