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
parser.add_argument('--max_episode_len', type=int, default=25, help='maximum episode length')
parser.add_argument('--policy_lr', type=float, default=0.01, help='learning rate for policies')
parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate for critics')
parser.add_argument('--alpha', type=float, default=0.0, help='policy entropy term coefficient')
parser.add_argument('--tau', type=float, default=0.05, help='target network smoothing coefficient')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size (default: 128)')
parser.add_argument('--hidden_dim', type=int, default=64, help='network hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, help='steps before training begins')
parser.add_argument('--target_update_interval', type=int, default=1, help='tagert network update interval')
parser.add_argument('--updates_per_step', type=int, default=1, help='network update frequency')
parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer')
parser.add_argument('--cuda', action='store_false', help='run on GPU (default: False)')
parser.add_argument('--render', action='store_true', help='render or not')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# TensorboardX
logdir = 'runs/MADDPG/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.scenario)
writer = SummaryWriter(logdir=logdir)

memory = ReplayMemory(args.replay_size)

# Load environment
scenario = scenarios.load(args.scenario + '.py').Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete_action_space=False)

obs_shape_list = [env.observation_space[i].shape[0] for i in range(env.n)]
action_shape_list = [env.action_space[i].shape[0] for i in range(env.n)]
trainers = []
for i in range(env.n):
    trainers.append(AgentTrainer(env.n, i, obs_shape_list, action_shape_list, args))

if os.path.exists('models/') and os.listdir('models/'):
    for t in trainers:
        t.load_model(env_name=args.scenario)
    print("Load models from files...")

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
        action_list = [agent.act(np.expand_dims(obs, axis=0)) for agent, obs in zip(trainers, obs_list)]

        # interact with the environment
        new_obs_list, reward_list, done_list, _ = env.step(deepcopy(action_list))
        if args.render:
            env.render()
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
                obs_batch, action_batch, reward_batch, next_obs_batch, _ = memory.sample(batch_size=args.batch_size)
                # Generate actions for sampled 'next_obs'
                next_action_list = [trainers[i].act(next_obs_batch[:,i]) for i in range(env.n)]
                next_action_batch = np.stack(next_action_list, axis=1)
                for i in range(env.n):
                    critic_loss, policy_loss = trainers[i].update_parameters((obs_batch, action_batch, reward_batch,
                        next_obs_batch, next_action_batch), args.batch_size, updates)

                    critic_losses.append(critic_loss)
                    policy_losses.append(policy_loss)

                    writer.add_scalar('loss/critic_{}'.format(i), critic_loss, updates)
                    writer.add_scalar('loss/policy_{}'.format(i), policy_loss, updates)
                updates += 1

    # logging episode stats
    for i in range(env.n):
        writer.add_scalar('reward/agent_{}'.format(i), episode_reward_per_agent[i], i_episode)
    writer.add_scalar('reward/total', episode_reward, i_episode)
    print("Episode: {}, total steps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
        step_within_episode, round(episode_reward, 2)))

    if i_episode > args.num_episodes:
        break

    # Test
    if i_episode % 100 == 0:
        episode_reward = 0.0
        step_within_episode = 0
        obs_list = env.reset()
        done = False

        while not done:
            action_list = [agent.act(np.expand_dims(obs, axis=0)) for agent, obs in zip(trainers, obs_list)]

            new_obs_list, reward_list, done_list, _ = env.step(deepcopy(action_list))
            env.render()
            step_within_episode += 1
            done = all(done_list)
            terminated = (step_within_episode >= args.max_episode_len)
            done = done or terminated

            obs_list = new_obs_list
            episode_reward += sum(reward_list)

for t in trainers:
    t.save_model(args.scenario)

env.close()