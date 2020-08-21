import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl import action_value
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction
from pfrl.initializers import init_chainer_default


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class DuelingDQN(nn.Module, StateQFunction):
    """Dueling Q-Network

    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu,
                 bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1)
            ]
        )

        self.a_stream = MLP(1024, n_actions, [512])
        self.v_stream = MLP(1024, 1, [512])

        self.conv_layers.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = ya.mean(dim=1, keepdim=True)
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)

