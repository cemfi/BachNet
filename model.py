import torch.nn
from torch.nn.functional import softmax, one_hot, selu

import data


class BachNet(torch.nn.Module):
    def __init__(self, batch_size, output_sizes, hidden_size, context_radius):
        super().__init__()
        self.batch_size = batch_size
        self.context_radius = context_radius
        self.output_sizes = output_sizes

        part_length = data.pitch_size + len(data.indices_parts)
        input_vector_length = (5 * context_radius + 1) * part_length + \
                              (2 * context_radius + 1) * len(data.indices_extra)

        self.fc_bass_1 = torch.nn.Linear(input_vector_length, hidden_size)
        self.fc_bass_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_bass_3 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_alto_1 = torch.nn.Linear(input_vector_length + 1 * part_length, hidden_size)
        self.fc_alto_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alto_3 = torch.nn.Linear(hidden_size, output_sizes)

        self.fc_tenor_1 = torch.nn.Linear(input_vector_length + 2 * part_length, hidden_size)
        self.fc_tenor_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_tenor_3 = torch.nn.Linear(hidden_size, output_sizes)

    def forward(self, inputs):
        inputs_bass = torch.cat([v.view(self.batch_size, -1) for v in inputs.values()], dim=1)

        outputs_bass = selu(self.fc_bass_1(inputs_bass))
        outputs_bass = selu(self.fc_bass_2(outputs_bass))
        outputs_bass = self.fc_bass_3(outputs_bass)

        prediction_bass = one_hot(torch.max(outputs_bass, dim=1)[1], self.output_sizes).float()
        inputs_alto = torch.cat([inputs_bass, prediction_bass], dim=1)

        outputs_alto = selu(self.fc_alto_1(inputs_alto))
        outputs_alto = selu(self.fc_alto_2(outputs_alto))
        outputs_alto = self.fc_alto_3(outputs_alto)

        prediction_alto = one_hot(torch.max(outputs_alto, dim=1)[1], self.output_sizes).float()
        inputs_tenor = torch.cat([inputs_alto, prediction_alto], dim=1)

        outputs_tenor = selu(self.fc_tenor_1(inputs_tenor))
        outputs_tenor = selu(self.fc_tenor_2(outputs_tenor))
        outputs_tenor = self.fc_tenor_3(outputs_tenor)

        return {
            'alto': softmax(outputs_alto, dim=1),
            'tenor': softmax(outputs_tenor, dim=1),
            'bass': softmax(outputs_bass, dim=1)
        }
