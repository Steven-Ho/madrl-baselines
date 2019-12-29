import argparse
import time
import datetime
import numpy as np 
import torch
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MADDPG Args')
parser.add_argument('--scenario', type=str, default='simple', help='name of the scenario script')

