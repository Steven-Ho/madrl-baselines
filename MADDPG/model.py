import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
