import torch.nn
from torch.nn.functional import relu, softmax


class BachNet(torch.nn.Module):
    def __init__(self, input_size, output_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes

        hidden_size = 300

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_sizes)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = relu(outputs)
        outputs = self.fc2(outputs)
        outputs = relu(outputs)
        outputs = self.fc3(outputs)

        return {
            'alto': softmax(outputs, dim=2),
            'tenor': softmax(outputs, dim=2),
            'bass': softmax(outputs, dim=2)
        }
