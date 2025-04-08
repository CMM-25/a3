import torch
import torch.nn as nn


class Value(nn.Module):
    def __init__(self,
                 num_obs,
                 hidden_dims=None,
                 activation='elu',
                 device='cpu',
                 **kwargs):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if kwargs:
            print("Value.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Value, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs

        # Value function
        layers = [nn.Linear(mlp_input_dim, hidden_dims[0]).to(device), activation]
        for la in range(len(hidden_dims)):
            if la == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[la], 1).to(device))
            else:
                layers.append(nn.Linear(hidden_dims[la], hidden_dims[la + 1]).to(device))
                layers.append(activation)
        self.value = nn.Sequential(*layers)

    def forward(self, input_x, mask=None):
        return self.value(input_x)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
