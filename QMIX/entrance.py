import argparse
import time
import datetime
import numpy as np 
import torch
import os
import itertools
from copy import deepcopy
from tensorboardX import SummaryWriter
from buffer import ReplayMemory
from train import AgentsTrainer
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

parser = argparse.ArgumentParser(description='PyTorch QMIX Args')
parser.add_argument('--scenario', type=str, default='simple_spread', help='name of the scenario script')
parser.add_argument('--num_episodes', type=int, default=60000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=25, help='maximum episode length')
parser.add_argument('--policy_lr', type=float, default=0.0001, help='learning rate for policies')
parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate for critics')
parser.add_argument('--alpha', type=float, default=0.0, help='policy entropy term coefficient')
parser.add_argument('--tau', type=float, default=0.05, help='target network smoothing coefficient')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 16)') # episodes
parser.add_argument('--hidden_dim', type=int, default=64, help='network hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, help='steps before training begins')
parser.add_argument('--target_update_interval', type=int, default=1, help='tagert network update interval')
parser.add_argument('--updates_per_step', type=int, default=1, help='network update frequency')
parser.add_argument('--replay_size', type=int, default=50000, help='maximum number of episodes of replay buffer')
parser.add_argument('--cuda', action='store_false', help='run on GPU (default: False)')
parser.add_argument('--render', action='store_true', help='render or not')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# TensorboardX
logdir = 'runs/QMIX/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.scenario)
writer = SummaryWriter(logdir=logdir)

# Load environment
scenario = scenarios.load(args.scenario + '.py').Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete_action_space=False)

obs_shape_list = [env.observation_space[i].shape[0] for i in range(env.n)]
action_shape_list = [env.action_space[i].shape[0] for i in range(env.n)]

# Homogeneity check
obs_shape = obs_shape_list[0]
action_shape = action_shape_list[0]
for i in range(len(obs_shape_list)):
    assert obs_shape_list[i] == obs_shape
    assert action_shape_list[i] == action_shape

memory = ReplayMemory(args.replay_size, args.max_episode_len, env.n, obs_shape, action_shape)

trainer = AgentsTrainer(env.n, obs_shape, action_shape, args)
total_numsteps = 0
updates = 0
t_start = time.time()

reward_bias = 30.
train_policy = False
for i_episode in itertools.count(1):
    episode_reward = 0.0 # sum of all agents
    episode_reward_per_agent = [0.0 for _ in range(env.n)] # reward list
    step_within_episode = 0

    obs_list = env.reset()
    done = False

    while not done:
        obs_array = np.asarray(obs_list)
        action_list = trainer.act(obs_array)
        action_list = list(action_list)

        # interact with the environment
        new_obs_list, reward_list, done_list, _ = env.step(deepcopy(action_list))

        if args.render:
            env.render()
        total_numsteps += 1
        step_within_episode += 1
        all_done = all(done_list)
        terminated = (step_within_episode >= args.max_episode_len)
        done = all_done or terminated

        # replay memory filling
        memory.push(np.asarray(obs_list), np.asarray(action_list), reward_list[0] + reward_bias, np.asarray(new_obs_list),
                    1. if (all_done and not terminated) else 0.)
        # memory.push(np.asarray(obs_list), np.asarray(action_list), reward_list[0], np.asarray(new_obs_list),
        #              1. if done else 0.)
        obs_list = new_obs_list

        episode_reward += sum(reward_list)
        for i in range(len(episode_reward_per_agent)):
            episode_reward_per_agent[i] += reward_list[i]

    memory.end_episode()
    trainer.reset()

    if i_episode > 1000:
        train_policy = True
    if len(memory) > args.batch_size:
        for _ in range(args.updates_per_step):
            obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, done_batch = memory.sample(args.batch_size)
            sample_batch = (obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, done_batch)
            critic_loss, policy_loss = trainer.update_parameters(sample_batch, args.batch_size, updates, train_policy)
            cl_mean = np.mean(np.asarray(critic_loss))
            pl_mean = np.mean(np.asarray(policy_loss))
            updates += 1
    else:
        cl_mean = 0. 
        pl_mean = 0. 

    writer.add_scalar('reward/total', episode_reward, i_episode)
    writer.add_scalar('loss/critic', cl_mean, i_episode)
    writer.add_scalar('loss/actor', pl_mean, i_episode)
    print("Episode: {}, total steps: {}, total episodes: {}, reward: {}".format(i_episode, total_numsteps,
        step_within_episode, round(episode_reward, 2)))

    if i_episode > args.num_episodes:
        break

env.close()