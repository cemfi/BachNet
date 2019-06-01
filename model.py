import torch.nn
from torch.nn.functional import softmax, one_hot, selu


class BachNet(torch.nn.Module):
    def __init__(self, input_size, output_sizes, hidden_size):
        super().__init__()
        self.output_sizes = output_sizes

        self.fc_bass_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_bass_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_bass_3 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_alto_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_alto_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alto_3 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_tenor_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_tenor_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_tenor_3 = torch.nn.Linear(hidden_size, output_sizes)

    def forward(self, inputs):
        inputs_all = torch.cat([v for v in inputs.values()], dim=2)

        outputs_bass = selu(self.fc_bass_1(inputs_all))
        outputs_bass = selu(self.fc_bass_2(outputs_bass))
        outputs_bass = self.fc_bass_3(outputs_bass)

        inputs['bass'] = one_hot(torch.max(outputs_bass, dim=2)[1], self.output_sizes).float()
        inputs_all = torch.cat([v for v in inputs.values()], dim=2)

        outputs_alto = selu(self.fc_alto_1(inputs_all))
        outputs_alto = selu(self.fc_alto_2(outputs_alto))
        outputs_alto = self.fc_alto_3(outputs_alto)

        inputs['alto'] = one_hot(torch.max(outputs_alto, dim=2)[1], self.output_sizes).float()
        inputs_all = torch.cat([v for v in inputs.values()], dim=2)

        outputs_tenor = selu(self.fc_tenor_1(inputs_all))
        outputs_tenor = selu(self.fc_tenor_2(outputs_tenor))
        outputs_tenor = self.fc_tenor_3(outputs_tenor)

        return {
            'alto': softmax(outputs_alto, dim=2),
            'tenor': softmax(outputs_tenor, dim=2),
            'bass': softmax(outputs_bass, dim=2)
        }
