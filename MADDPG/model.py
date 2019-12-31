import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_outputs):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init)

    def forward(self, inputs, output_activation=None):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        if output_activation is 'relu':
            x = F.relu(x)
        elif output_activation is 'softmax':
            x = F.softmax(x)

        return x

# class CategoricalPolicy(nn.Module):
#     def __init__(self, num_inputs, action_dims, hidden_dim):
#         super(CategoricalPolicy, self).__init__()

#         self.l1 = nn.Linear(num_inputs, hidden_dim)
#         self.l2 = nn.Linear(hidden_dim, hidden_dim)
#         self.logits = nn.Linear(hidden_dim, action_dims)

#         self.apply(weights_init)

#     def forward(self, state):
#         x = F.relu(self.l1(state))
#         x = F.relu(self.l2(x))
#         logits = self.logits(x)
#         prob = F.softmax(logits)

#         return logits, prob

#     def sample(self, state):
#         logits, prob = self.forward(state)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2.)

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

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

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)

        return super(GaussianPolicy, self).to(device)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.l1 = nn.Linear(num_actions + num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init)

    def forward(self, states, actions):
        x = torch.cat([states, actions], 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x