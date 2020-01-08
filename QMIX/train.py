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
    def __init__(self, num_agents, obs_shape, action_shape, args):
        self.na = num_agents
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.args = args

        self.target_update_interval = args.target_update_interval
        self.tau = args.tau
        self.alpha = 0.
        self.gamma = args.gamma
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # Suppose agents are homogeneous
        # Use critics and actors with shared parameters
        # Therefore agent id (onehot vector) is needed
        self.critics = RNNQNetwork(num_agents + obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.critics_target = RNNQNetwork(num_agents + obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.critics_optim = Adam(self.critics.parameters(), lr=args.critic_lr)
        hard_update(self.critics_target, self.critics)

        self.actors = RNNGaussianPolicy(num_agents + obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.actors_target = RNNGaussianPolicy(num_agents + obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.actors_optim = Adam(self.actors.parameters(), lr=args.policy_lr)
        hard_update(self.actors_target, self.actors)

        self.qmix_net = QMIXNetwork(num_agents, args.hidden_dim, obs_shape * num_agents).to(device=self.device)
        self.qmix_net_target = QMIXNetwork(num_agents, args.hidden_dim, obs_shape * num_agents).to(device=self.device)
        self.qmix_net_optim = Adam(self.qmix_net.parameters(), lr=args.critic_lr)
        hard_update(self.qmix_net_target, self.qmix_net)

    def act(self, obs, eval=False):
        obs = self.make_input(obs)
        obs = torch.FloatTensor(obs).to(device=self.device)
        if eval:
            _, _, actions = self.actors_target.sample(obs)
        else:
            actions, _, _ = self.actors_target.sample(obs)

        return actions.detach().cpu().numpy()
    
    def make_input(self, obs):
        if len(obs.shape) == 3:
            obs = obs.reshape(-1, self.obs_shape)
        identity = np.eye(self.na)
        num_episodes = int(obs.shape[0] / self.na)
        identity = np.tile(identity, (num_episodes, 1))
        enhanced_obs = np.concatenate((obs, identity), axis=1)

        return enhanced_obs

    def reset(self):
        self.actors.reset()
        self.actors_target.reset()
        self.critics.reset()
        self.critics_target.reset()

    def update_parameters(self, samples, batch_size, updates):
        pass