import torch.nn as nn
import torch.nn.functional as F


# Model definition
class MyModel(nn.Module):
    def __init__(self, hidden_layer_neurons=100):
        super(MyModel, self).__init__()
        self.hidden_layer_neurons = hidden_layer_neurons
        self.fc1 = nn.Linear(28 * 28, self.hidden_layer_neurons)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_layer_neurons, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
