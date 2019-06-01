import torch.nn
from torch.nn.functional import relu, softmax


class BachNet(torch.nn.Module):
    def __init__(self, input_size, output_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes

        hidden_size = 300

        self.fc_global_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_global_2 = torch.nn.Linear(hidden_size, hidden_size)

        self.fc_bass_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_bass_2 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_alto_1 = torch.nn.Linear(hidden_size + 1 * output_sizes, hidden_size)
        self.fc_alto_2 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_tenor_1 = torch.nn.Linear(hidden_size + 2 * output_sizes, hidden_size)
        self.fc_tenor_2 = torch.nn.Linear(hidden_size, output_sizes)

    def forward(self, inputs):
        outputs = relu(self.fc_global_1(inputs))
        outputs = relu(self.fc_global_2(outputs))
        
        outputs_bass = relu(self.fc_bass_1(outputs))
        outputs_bass = self.fc_bass_2(outputs_bass)

        outputs_alto = relu(self.fc_alto_1(torch.cat((outputs, outputs_bass), dim=2)))
        outputs_alto = self.fc_alto_2(outputs_alto)

        outputs_tenor = relu(self.fc_tenor_1(torch.cat((outputs, outputs_bass, outputs_alto), dim=2)))
        outputs_tenor = self.fc_tenor_2(outputs_tenor)

        return {
            'alto': softmax(outputs_alto, dim=2),
            'tenor': softmax(outputs_tenor, dim=2),
            'bass': softmax(outputs_bass, dim=2)
        }
