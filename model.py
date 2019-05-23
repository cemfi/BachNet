import torch.utils
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BachBase(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_hidden_layers, dropout=0.5):
        super(BachBase, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.gru1 = torch.nn.GRU(input_size=input_dims, num_layers=1, hidden_size=hidden_dims, bidirectional=True)
        self.gru2 = torch.nn.GRU(input_size=hidden_dims * 2, num_layers=1, hidden_size=hidden_dims, bidirectional=True)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc1b = torch.nn.Linear(hidden_dims * 2, output_dims)
        self.fc2a = torch.nn.Linear(hidden_dims * 2 + output_dims, output_dims)
        self.fc3t = torch.nn.Linear(hidden_dims * 2 + output_dims * 2, output_dims)

    def init_hidden(self, n_seqs):
        """Initializes hidden state"""
        weight = next(self.parameters()).data
        return weight.new(self.num_hidden_layers, n_seqs, self.hidden_dims).zero_()


class BachNet(BachBase):
    def forward(self, x, hidden):
        out, hidden = self.gru1(x, hidden)
        out, hidden = self.gru2(out, hidden)
        out = self.dropout1(out)
        out_1B = self.fc1b(out)
        out_plusB= torch.cat((out, out_1B), 2)
        out_2A = self.fc2a(out_plusB)
        out_plusB_plusA = torch.cat((out_plusB, out_2A), 2)
        out_3T = self.fc3t(out_plusB_plusA)
        return out_1B, out_2A, out_3T, hidden


class AnalysisNet(BachBase):
    def forward(self, x, hidden):
        print(self.layer_neuron_pair)
        out, hidden = self.gru1(x, hidden)
        neurons = []
        neurons.append(out.permute(2, 1, 0)[:, 0, :])
        out, hidden = self.gru2(out, hidden)
        neurons.append(out.permute(2, 1, 0)[:, 0, :])
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        neurons.append(out.permute(2, 1, 0)[:, 0, :])
        out = self.fc2(out)
        return out, hidden, neurons


class PlotNeuronNet(BachBase):
    def __init__(self, input_dims, hidden_dims, output_dims, num_hidden_layers, dropout=0.5, neuron=1):
        super(PlotNeuronNet3, self).__init__(input_dims, hidden_dims, output_dims, num_hidden_layers, dropout)
        self.neuron = neuron

    def forward(self, x, hidden):
        out, hidden = self.gru1(x, hidden)
        out, hidden = self.gru2(out, hidden)
        neuron = out[:, 0, self.neuron]
        neuron -= torch.min(neuron)   # shift to min zero
        neuron /= torch.max(neuron)   # scale to max one
        plt.plot(neuron.detach().numpy())
        plt.draw()
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden
