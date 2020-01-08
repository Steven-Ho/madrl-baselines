import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class RNNGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(RNNGaussianPolicy, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        # intermediate state, set to None at the beginning of an episode
        self.h = None

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2.)   

    def forward(self, state):
        x = F.relu(self.l1(state))
        if self.h is None:
            h_out = self.rnn(x)
        else:
            h_out = self.rnn(x, self.h)
        self.h = h_out

        mean = self.mean(h_out)
        log_std = self.log_std(h_out)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def _get_hidden_state(self):
        return self.h

    def reset(self):
        self.h = None

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_p = normal.log_prob(x)

        # enforcing action bound
        log_p -= torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
        log_p = log_p.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_p, mean     

class RNNQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(RNNQNetwork, self).__init__()

        self.l1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

        # intermediate state, set to None at the beginning of an episode
        self.h = None

        self.apply(weights_init)

    def forward(self, states, actions):
        x = torch.cat([states, actions], 1)

        x = F.relu(self.l1(x))
        if self.h is None:
            h_out = self.rnn(x)
        else:
            h_out = self.rnn(x, self.h)
        self.h = h_out
        x = self.l2(h_out)

        return x

    def _get_hidden_state(self):
        return self.h

    def reset(self):
        self.h = None

class QMIXNetwork(nn.Module):
    def __init__(self, num_agents, hidden_dim, total_obs_dim):
        super(QMIXNetwork, self).__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Linear(total_obs_dim, num_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(total_obs_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(total_obs_dim, hidden_dim)
        self.hyper_b2_l1 = nn.Linear(total_obs_dim, hidden_dim)
        self.hyper_b2_l2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init)

    def forward(self, q_values, total_obs):
        w1 = torch.abs(self.hyper_w1(total_obs))
        b1 = self.hyper_b1(total_obs)
        w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)
        b1 = b1.reshape(-1, 1, self.hidden_dim)

        x = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(total_obs))
        b2 = self.hyper_b2_l2(F.relu(self.hyper_b2_l1(total_obs)))
        w2 = w2.reshape(-1, self.hidden_dim, 1)
        b2 = b2.reshape(-1, 1, 1)

        x = torch.bmm(x, w2) + b2

        return x