import numpy as np 
import random
import torch
from functools import reduce
from torch.optim import Adam
from utils import soft_update, hard_update
from MADDPG.model import GaussianPolicy, QNetwork
from MADDPG.buffer import ReplayMemory

class AgentTrainer(object):
    def __init__(self, num_agents, agent_index, obs_shape_list, action_shape_list, args):
        self.na = num_agents
        self.index = agent_index
        self.args = args

        self.target_update_interval = args.target_update_interval
        self.tau = args.tau
        self.alpha = 0.
        self.gamma = args.gamma
        self.device = torch.device("cuda" if args.cuda else "cpu")

        num_inputs = reduce((lambda x,y: x+y), obs_shape_list)
        num_actions = reduce((lambda x,y: x+y), action_shape_list)

        self.critic = QNetwork(num_inputs, num_actions, args.hidden_dim).to(device=self.device)
        self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(obs_shape_list[self.index], action_shape_list[self.index], args.hidden_dim).to(device=self.device)
        self.policy_target = GaussianPolicy(obs_shape_list[self.index], action_shape_list[self.index], args.hidden_dim).to(device=self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.policy_lr)
        hard_update(self.policy_target, self.policy)

    def act(self, obs, eval=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if eval:
            _, _, action = self.policy_target.sample(obs)
        else:
            action, _, _ = self.policy_target.sample(obs)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch
        obs_batch, action_batch, reward_batch, next_obs_batch = memory.sample(batch_size=batch_size)
        # Leave the correspondent data
        obs_batch = obs_batch[:,self.index]
        action_batch = action_batch[:,self.index]
        reward_batch = reward_batch[:,self.index]
        next_obs_batch = next_obs_batch[:,self.index]
        
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        with torch.no_grad():
            next_action, next_log_p, _ = self.policy_target.sample(next_obs_batch)
            next_q = self.critic_target(next_obs_batch, next_action) - self.alpha * next_log_p
            td_q = reward_batch + self.gamma * next_q

        q = self.critic(obs_batch, action_batch)
        q_loss = F.mse_loss(q, td_q)

        a, log_p, _ = self.policy.sample(obs_batch)
        q_p = self.critic(obs_batch, a)
        policy_loss = (self.alpha * log_p - q_p).mean()

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)