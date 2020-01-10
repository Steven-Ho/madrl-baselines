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

EPS = 1e-2

class AgentsTrainer(object):
    def __init__(self, num_agents, obs_shape, action_shape, args):
        self.na = num_agents
        self.obs_shape = obs_shape
        self.enhanced_obs_shape = obs_shape + num_agents
        self.action_shape = action_shape
        self.args = args

        self.target_update_interval = args.target_update_interval
        self.tau = args.tau
        self.alpha = 0.
        self.gamma = args.gamma
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # Hidden states used in policy rollouts
        self.actor_h = torch.zeros(num_agents, args.hidden_dim).to(device=self.device)

        # Suppose agents are homogeneous
        # Use critics and actors with shared parameters
        # Therefore agent id (onehot vector) is needed
        self.critics = RNNQNetwork(self.enhanced_obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.critics_target = RNNQNetwork(self.enhanced_obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.critics_optim = Adam(self.critics.parameters(), lr=args.critic_lr)
        hard_update(self.critics_target, self.critics)

        self.actors = RNNGaussianPolicy(self.enhanced_obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.actors_target = RNNGaussianPolicy(self.enhanced_obs_shape, action_shape, args.hidden_dim).to(device=self.device)
        self.actors_optim = Adam(self.actors.parameters(), lr=args.policy_lr)
        hard_update(self.actors_target, self.actors)

        self.qmix_net = QMIXNetwork(num_agents, 16, self.enhanced_obs_shape * num_agents).to(device=self.device)
        self.qmix_net_target = QMIXNetwork(num_agents, 16, self.enhanced_obs_shape * num_agents).to(device=self.device)
        self.qmix_net_optim = Adam(self.qmix_net.parameters(), lr=args.critic_lr)
        hard_update(self.qmix_net_target, self.qmix_net)

    def act(self, obs, eval=False):
        obs = self.make_input(obs)
        obs = torch.FloatTensor(obs).to(device=self.device)
        if eval:
            _, _, actions, self.actor_h = self.actors_target.sample(obs, self.actor_h)
        else:
            actions, _, _, self.actor_h = self.actors_target.sample(obs, self.actor_h)

        return actions.detach().cpu().numpy()
    
    def make_input(self, obs):
        shape = list(obs.shape)
        shape[-1] = self.enhanced_obs_shape
        if len(shape) >= 3:
            obs = obs.reshape(-1, self.obs_shape)
        identity = np.eye(self.na)
        n = int(obs.shape[0] / self.na) # number of entries in total
        identity = np.tile(identity, (n, 1))
        enhanced_obs = np.concatenate((obs, identity), axis=1)
        enhanced_obs = enhanced_obs.reshape(shape)

        return enhanced_obs

    def reset(self):
        self.actor_h = torch.zeros(self.na, self.args.hidden_dim).to(device=self.device)

    def update_parameters(self, samples, batch_size, updates, train_policy):
        obs_batch, action_batch, reward_batch, obs_next_batch, mask_batch, done_batch = samples
        obs_batch = self.make_input(obs_batch)
        obs_next_batch = self.make_input(obs_next_batch)

        obs_batch = torch.FloatTensor(obs_batch).to(device=self.device)
        action_batch = torch.FloatTensor(action_batch).to(device=self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(device=self.device)
        obs_next_batch = torch.FloatTensor(obs_next_batch).to(device=self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(device=self.device)
        done_batch = torch.FloatTensor(done_batch).to(device=self.device)

        max_episode_len = obs_batch.shape[1]
        
        actors_h = torch.zeros(batch_size * self.na, self.args.hidden_dim).to(device=self.device)
        critics_1_h = torch.zeros(batch_size * self.na, self.args.hidden_dim).to(device=self.device)
        critics_2_h = torch.zeros(batch_size * self.na, self.args.hidden_dim).to(device=self.device)
        actors_target_h = torch.zeros(batch_size * self.na, self.args.hidden_dim).to(device=self.device)
        critics_target_h = torch.zeros(batch_size * self.na, self.args.hidden_dim).to(device=self.device)

        cl = []
        pl = []
        for i in range(max_episode_len):
            # train in time order
            obs_slice = obs_batch[:,i].squeeze().reshape(-1, self.enhanced_obs_shape)
            total_obs_slice = obs_batch[:,i].squeeze().reshape(-1, self.na * self.enhanced_obs_shape)
            action_slice = action_batch[:,i].squeeze().reshape(-1, self.action_shape)
            reward_slice = reward_batch[:,i].squeeze().reshape(-1)
            obs_next_slice = obs_next_batch[:,i].squeeze().reshape(-1, self.enhanced_obs_shape)
            total_obs_next_slice = obs_next_batch[:,i].squeeze().reshape(-1, self.na * self.enhanced_obs_shape)
            mask_slice = mask_batch[:,i].squeeze().reshape(-1)
            done_slice = done_batch[:,i].squeeze().reshape(-1)

            if mask_slice.sum().cpu().numpy() < EPS:
                break

            with torch.no_grad():
                action_next, log_p_next, _, actors_target_h = self.actors_target.sample(obs_next_slice, actors_target_h)
                qs_next, critics_target_h = self.critics_target(obs_next_slice, action_next, critics_target_h)# - self.alpha * log_p_next
                qs_next = qs_next.reshape(-1, self.na)
                q_next = self.qmix_net_target(qs_next, total_obs_next_slice)
                td_q = (reward_slice + self.gamma * (1. - done_slice) * q_next) * mask_slice

            temp_h = critics_1_h
            qs, critics_1_h = self.critics(obs_slice, action_slice, critics_1_h)
            q = self.qmix_net(qs.reshape(-1, self.na), total_obs_slice) * mask_slice
            q_loss = ((q - td_q)**2).sum() / mask_slice.sum()

            a, log_p, _, actors_h = self.actors.sample(obs_slice, actors_h)
            qs_a, critics_2_h = self.critics(obs_slice, a, critics_2_h)
            q_a = self.qmix_net(qs_a.reshape(-1, self.na), total_obs_slice) * mask_slice
            p_loss = - q_a.sum() / mask_slice.sum() # entropy term removed

            self.critics_optim.zero_grad()
            self.qmix_net_optim.zero_grad()
            q_loss.backward(retain_graph=True)
            self.critics_optim.step()
            self.qmix_net_optim.step()

            qs, _ = self.critics(obs_slice, action_slice, temp_h)
            q = self.qmix_net(qs.reshape(-1, self.na), total_obs_slice) * mask_slice
            q_loss_trained = ((q - td_q)**2).sum() / mask_slice.sum()            

            if train_policy:
                self.actors_optim.zero_grad()
                p_loss.backward(retain_graph=True)
                self.actors_optim.step()

            cl.append(q_loss.detach().cpu().numpy())
            pl.append(p_loss.detach().cpu().numpy())
            if updates % self.target_update_interval == 0:
                soft_update(self.critics_target, self.critics, self.tau)
                soft_update(self.actors_target, self.actors, self.tau)
                soft_update(self.qmix_net_target, self.qmix_net, self.tau)

        return cl, pl            