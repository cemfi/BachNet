import torch.utils
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BachBase(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_hidden_layers, dropout=0.5):
        super(BachBase, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.gru = torch.nn.GRU(input_size=input_dims, num_layers=num_hidden_layers, hidden_size=hidden_dims, bidirectional=True, dropout=dropout)
        self.fc1 = torch.nn.Linear(hidden_dims * 2, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, output_dims)

    def init_hidden(self, n_seqs):
        """Initializes hidden state"""
        weight = next(self.parameters()).data
        return weight.new(self.num_hidden_layers * 2, n_seqs, self.hidden_dims).zero_()


class BachNet3(BachBase):
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden


class AnalysisNet3(BachBase):
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        neuron_all_steps = out.permute(2, 1, 0)[:, 0, :]
        out = F.relu(self.fc1(out))
        # out = self.dropout(out)
        # out = self.bn1(out)
        out = self.fc2(out)
        return out, hidden, neuron_all_steps


class PlotNeuronNet3(BachBase):
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        plt.plot(out[:, 0, 11].detach().numpy())
        plt.show()
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden
