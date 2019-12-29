import argparse
import time
import datetime
import numpy as np 
import torch
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MADDPG Args')
parser.add_argument('--scenario', type=str, default='simple', help='name of the scenario script')
parser.add_argument('--policy_lr', type=float, default=0.0003, help='learning rate for policies')
parser.add_argument('--critic_lr', type=float, default=0.0003, help='learning rate for critics')
parser.add_argument('--alpha', type=float, default=0.0, help='policy entropy term coefficient')
parser.add_argument('--tau', type=float, default=0.05, help='target network smoothing coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--hidden_dim', type=int, default=256, help='network hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, help='steps before training begins')
parser.add_argument('--target_update_interval', type=int, default=1, help='tagert network update interval')
parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer')
parser.add_argument('--cuda', action='store_true', help='run on GPU (default: False)')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# TensorboardX
