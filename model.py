import torch.nn
from torch.nn.functional import one_hot, selu

import data


class BachNetBase(torch.nn.Module):
    def __init__(self, hidden_size, context_radius, dropout=0.5):
        super(BachNetBase, self).__init__()
        self.output_size = data.pitch_size + len(data.indices_parts)
        input_size = (5 * context_radius + 1) * self.output_size + \
                     (2 * context_radius + 1) * len(data.indices_extra)

        self.dropout = torch.nn.Dropout(dropout)

        self.fc_bass_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_bass_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_bass_3 = torch.nn.Linear(hidden_size, self.output_size)

        self.fc_alto_1 = torch.nn.Linear(input_size + 1 * self.output_size, hidden_size)
        self.fc_alto_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alto_3 = torch.nn.Linear(hidden_size, self.output_size)

        self.fc_tenor_1 = torch.nn.Linear(input_size + 2 * self.output_size, hidden_size)
        self.fc_tenor_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_tenor_3 = torch.nn.Linear(hidden_size, self.output_size)

    def forward(self, inputs):
        raise NotImplementedError()


class BachNetTraining(BachNetBase):
    def forward(self, inputs):
        batch_size = inputs['soprano'].shape[0]

        inputs_bass = torch.cat([v.view(batch_size, -1) for v in inputs.values()], dim=1)

        outputs_bass = selu(self.fc_bass_1(inputs_bass))
        outputs_bass = self.dropout(outputs_bass)
        outputs_bass = selu(self.fc_bass_2(outputs_bass))
        outputs_bass = self.dropout(outputs_bass)
        outputs_bass = self.fc_bass_3(outputs_bass)

        prediction_bass = one_hot(torch.max(outputs_bass, dim=1)[1], self.output_size).float()
        inputs_alto = torch.cat([inputs_bass, prediction_bass], dim=1)

        outputs_alto = selu(self.fc_alto_1(inputs_alto))
        outputs_alto = self.dropout(outputs_alto)
        outputs_alto = selu(self.fc_alto_2(outputs_alto))
        outputs_alto = self.dropout(outputs_alto)
        outputs_alto = self.fc_alto_3(outputs_alto)

        prediction_alto = one_hot(torch.max(outputs_alto, dim=1)[1], self.output_size).float()
        inputs_tenor = torch.cat([inputs_alto, prediction_alto], dim=1)

        outputs_tenor = selu(self.fc_tenor_1(inputs_tenor))
        outputs_tenor = self.dropout(outputs_tenor)
        outputs_tenor = selu(self.fc_tenor_2(outputs_tenor))
        outputs_tenor = self.dropout(outputs_tenor)
        outputs_tenor = self.fc_tenor_3(outputs_tenor)

        return {
            'alto': outputs_alto,
            'tenor': outputs_tenor,
            'bass': outputs_bass
        }


class BachNetInference(BachNetBase):
    def forward(self, inputs):
        inputs_bass = torch.cat([v.view(self.batch_size, -1) for v in inputs.values()], dim=1)

        outputs_bass = selu(self.fc_bass_1(inputs_bass))
        outputs_bass = self.dropout(outputs_bass)
        outputs_bass = selu(self.fc_bass_2(outputs_bass))
        outputs_bass = self.dropout(outputs_bass)
        outputs_bass = self.fc_bass_3(outputs_bass)

        prediction_bass = one_hot(torch.max(outputs_bass, dim=1)[1], self.output_size).float()
        inputs_alto = torch.cat([inputs_bass, prediction_bass], dim=1)

        outputs_alto = selu(self.fc_alto_1(inputs_alto))
        outputs_alto = self.dropout(outputs_alto)
        outputs_alto = selu(self.fc_alto_2(outputs_alto))
        outputs_alto = self.dropout(outputs_alto)
        outputs_alto = self.fc_alto_3(outputs_alto)

        prediction_alto = one_hot(torch.max(outputs_alto, dim=1)[1], self.output_size).float()
        inputs_tenor = torch.cat([inputs_alto, prediction_alto], dim=1)

        outputs_tenor = selu(self.fc_tenor_1(inputs_tenor))
        outputs_tenor = self.dropout(outputs_tenor)
        outputs_tenor = selu(self.fc_tenor_2(outputs_tenor))
        outputs_tenor = self.dropout(outputs_tenor)
        outputs_tenor = self.fc_tenor_3(outputs_tenor)

        prediction_tenor = one_hot(torch.max(outputs_tenor, dim=1)[1], self.output_size).float()

        return {
            'alto': prediction_alto,
            'tenor': prediction_tenor,
            'bass': prediction_bass
        }
