import numpy as np 
import random
import torch
import os
from functools import reduce
from torch.optim import Adam
import torch.nn.functional as F
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
        obs = torch.FloatTensor(obs).to(self.device)
        if eval:
            _, _, action = self.policy_target.sample(obs)
        else:
            action, _, _ = self.policy_target.sample(obs)
        
        return action.squeeze().detach().cpu().numpy()

    def update_parameters(self, samples, batch_size, updates):
        # Sample a batch
        obs_batch, action_batch, reward_batch, next_obs_batch, next_action_batch = samples
        # Leave the correspondent data
        obs_batch_i = obs_batch[:,self.index]
        action_batch_i = action_batch[:,self.index]
        reward_batch_i = reward_batch[:,self.index]
        next_obs_batch_i = next_obs_batch[:,self.index]
        # Reshape
        obs_batch = np.reshape(obs_batch, (batch_size, -1))
        action_batch = np.reshape(action_batch, (batch_size, -1))
        next_obs_batch = np.reshape(next_obs_batch, (batch_size, -1))
        # Move to the device designated
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        next_actions = torch.FloatTensor(next_action_batch).to(self.device)
        all_a = torch.FloatTensor(next_action_batch).to(self.device)
        obs_batch_i = torch.FloatTensor(obs_batch_i).to(self.device)
        action_batch_i = torch.FloatTensor(action_batch_i).to(self.device)
        reward_batch_i = torch.FloatTensor(reward_batch_i).to(self.device)
        reward_batch_i = reward_batch_i.unsqueeze(-1)
        next_obs_batch_i = torch.FloatTensor(next_obs_batch_i).to(self.device)

        with torch.no_grad():
            next_action, next_log_p, _ = self.policy_target.sample(next_obs_batch_i)
            next_actions[:,self.index] = next_action
            next_actions = torch.reshape(next_actions, (batch_size, -1))
            next_q = self.critic_target(next_obs_batch, next_actions) - self.alpha * next_log_p
            td_q = reward_batch_i + self.gamma * next_q

        q = self.critic(obs_batch, action_batch)
        q_loss = F.mse_loss(q, td_q)

        a, log_p, _ = self.policy.sample(obs_batch_i)
        all_a[:,self.index] = a
        all_a = torch.reshape(all_a, (batch_size, -1))
        q_p = self.critic(obs_batch, all_a)
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

        return q_loss, policy_loss

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/maddpg_actor_{}_{}_{}".format(env_name, suffix, self.index)
        if critic_path is None:
            critic_path = "models/maddpg_critic_{}_{}_{}".format(env_name, suffix, self.index)

        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)        
    
    # Load model parameters
    def load_model(self, actor_path=None, critic_path=None, env_name=None, suffix=""):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        elif env_name is not None:
            actor_path = "models/maddpg_actor_{}_{}_{}".format(env_name, suffix, self.index)
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        elif env_name is not None:
            critic_path = "models/maddpg_critic_{}_{}_{}".format(env_name, suffix,self.index)
            self.critic.load_state_dict(torch.load(critic_path))

        hard_update(self.critic_target, self.critic)
        hard_update(self.policy_target, self.policy)