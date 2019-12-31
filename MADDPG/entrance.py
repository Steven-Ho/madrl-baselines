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
from train import AgentTrainer
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

parser = argparse.ArgumentParser(description='PyTorch MADDPG Args')
parser.add_argument('--scenario', type=str, default='simple_spread', help='name of the scenario script')
parser.add_argument('--num_episodes', type=int, default=60000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=50, help='maximum episode length')
parser.add_argument('--policy_lr', type=float, default=0.0003, help='learning rate for policies')
parser.add_argument('--critic_lr', type=float, default=0.0003, help='learning rate for critics')
parser.add_argument('--alpha', type=float, default=0.0, help='policy entropy term coefficient')
parser.add_argument('--tau', type=float, default=0.05, help='target network smoothing coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 128)')
parser.add_argument('--hidden_dim', type=int, default=256, help='network hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, help='steps before training begins')
parser.add_argument('--target_update_interval', type=int, default=1, help='tagert network update interval')
parser.add_argument('--updates_per_step', type=int, default=1, help='network update frequency')
parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer')
parser.add_argument('--cuda', action='store_false', help='run on GPU (default: False)')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# TensorboardX
logdir = 'runs/{}_MADDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.scenario)
writer = SummaryWriter(logdir=logdir)

memory = ReplayMemory(args.replay_size)

# Load environment
scenario = scenarios.load(args.scenario + '.py').Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

obs_shape_list = [env.observation_space[i].shape[0] for i in range(env.n)]
action_shape_list = [env.action_space[i].shape[0] for i in range(env.n)]
trainers = []
for i in range(env.n):
    trainers.append(AgentTrainer(env.n, i, obs_shape_list, action_shape_list, args))

total_numsteps = 0
updates = 0
t_start = time.time()

for i_episode in itertools.count(1):
    episode_reward = 0.0 # sum of all agents
    episode_reward_per_agent = [0.0 for _ in range(env.n)] # reward list
    step_within_episode = 0

    obs_list = env.reset()
    done = False

    while not done:
        # TODO: substitute the actions with random ones when starts up
        action_list = [agent.act(obs) for agent, obs in zip(trainers, obs_list)]

        # interact with the environment
        new_obs_list, reward_list, done_list, _ = env.step(deepcopy(action_list))
        total_numsteps += 1
        step_within_episode += 1
        done = all(done_list)
        terminated = (step_within_episode >= args.max_episode_len)
        done = done or terminated

        # replay memory filling
        memory.push((obs_list, action_list, reward_list, new_obs_list, done_list))
        obs_list = new_obs_list

        episode_reward += sum(reward_list)
        for i in range(len(episode_reward_per_agent)):
            episode_reward_per_agent[i] += reward_list[i]

        if len(memory) > 2 * args.batch_size:
            for _ in range(args.updates_per_step):
                critic_losses = []
                policy_losses = []
                for i in range(env.n):
                    critic_loss, policy_loss = trainers[i].update_parameters(memory, args.batch_size, updates)

                    critic_losses.append(critic_loss)
                    policy_losses.append(policy_loss)

                    writer.add_scalar('loss/critic_{}'.format(i), critic_loss, updates)
                    writer.add_scalar('loss/policy_{}'.format(i), policy_loss, updates)
                    updates += 1

    # logging episode stats
    for i in range(env.n):
        writer.add_scalar('reward/agent_{}'.format(i), episode_reward_per_agent[i], i_episode)
    writer.add_scalar('reward/total', episode_reward, i_episode)
    print("Episode: {}, total steps: {}, total episodes: {}, reward: {}".format(i_episode, total_numsteps,
        step_within_episode, episode_reward))