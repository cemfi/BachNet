import math

import torch.nn
from torch.nn.functional import one_hot

from data import part_size, indices_extra


class BachNetBase(torch.nn.Module):
    def __init__(self, hidden_size, context_radius, dropout=0.5, full_context_number=2, half_context_number=2):
        super(BachNetBase, self).__init__()
        input_full_context_size = full_context_number * (2 * context_radius + 1) * part_size
        input_half_context_size = (half_context_number * context_radius) * part_size
        input_size = input_full_context_size + input_half_context_size + \
                     (2 * context_radius + 1) * len(indices_extra)

        self.bass = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, part_size),
        )

        self.alto = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, part_size),
        )

        self.tenor = torch.nn.Sequential(
            torch.nn.Linear(input_size + part_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, part_size),
        )

    def forward(self, inputs):
        raise NotImplementedError()


class BachNetTrainingContinuo(BachNetBase):
    def forward(self, inputs):
        batch_size = inputs['soprano'].shape[0]
        inputs_bass = torch.cat([inputs[k].view(batch_size, -1) for k in ['soprano', 'bass', 'extra']], dim=1)

        outputs_bass = self.bass(inputs_bass)
        # prediction_bass = one_hot(torch.max(outputs_bass, dim=1)[1], part_size).float()

        return {
            'bass': outputs_bass
        }


class BachNetTrainingMiddleParts(BachNetBase):
    def forward(self, inputs):
        batch_size = inputs['soprano'].shape[0]
        inputs_alto = torch.cat(
            [inputs[k].view(batch_size, -1) for k in ['soprano', 'alto', 'tenor', 'bass_withcontext', 'extra']], dim=1)

        outputs_alto = self.alto(inputs_alto)
        prediction_alto = one_hot(torch.max(outputs_alto, dim=1)[1], part_size).float()

        inputs_tenor = torch.cat([inputs_alto, prediction_alto], dim=1)
        outputs_tenor = self.tenor(inputs_tenor)

        return {
            'alto': outputs_alto,
            'tenor': outputs_tenor
        }


class BachNetInferenceContinuo(BachNetBase):
    def __init__(self, num_candidates, *args, **kwargs):
        super(BachNetInferenceContinuo, self).__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        self.weight_bass = 1
        self.weight_alto = 1
        self.weight_tenor = 1

    def set_part_weights(self, loss_bass):#, loss_alto, loss_tenor):
        maximum = 1 / loss_bass#, 1 / loss_alto, 1 / loss_tenor)
        self.weight_bass = 1 / loss_bass / maximum
        #self.weight_alto = 1 / loss_alto / maximum
        #self.weight_tenor = 1 / loss_tenor / maximum

    def forward(self, inputs):
        num_parts = 3
        results = torch.zeros((self.num_candidates, 1 + num_parts))  # [[Candidate index], [[ProbAcc, PitchB, PitchA, PitchT]]

        # Bass #################################################################
        inputs_bass = torch.cat(
            [inputs[k].view(1, -1) for k in ['soprano', 'bass', 'extra']],
            dim=1).squeeze()  # !!! SQUEEZED !!!

        outputs_bass = self.bass(inputs_bass)

        log_probabilities, pitches = torch.sort(torch.log(torch.softmax(outputs_bass, dim=0)), dim=0, descending=True)
        log_probabilities += math.log(self.weight_bass)
        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches[:self.num_candidates]

        return results


class BachNetInferenceMiddleParts(BachNetBase):
    def __init__(self, num_candidates, *args, **kwargs):
        super(BachNetInferenceMiddleParts, self).__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        self.weight_alto = 1
        self.weight_tenor = 1

    def set_part_weights(self, loss_alto, loss_tenor):
        maximum = max(1 / loss_alto, 1 / loss_tenor)
        self.weight_alto = 1 / loss_alto / maximum
        self.weight_tenor = 1 / loss_tenor / maximum

    def forward(self, inputs):
        num_parts = 3
        results = torch.zeros((self.num_candidates, 1 + num_parts))  # [[Candidate index], [[ProbAcc, PitchB, PitchA, PitchT]]

        # Alto #################################################################
        inputs_alto = torch.cat([inputs[k].view(1, -1) for k in ['soprano', 'alto', 'tenor', 'bass_withcontext', 'extra']],
            dim=1).squeeze()  # !!! SQUEEZED !!!
        outputs_alto = self.alto(inputs_alto)

        log_probabilities, pitches = torch.sort(torch.log(torch.softmax(outputs_alto, dim=0)), dim=0, descending=True)
        log_probabilities += math.log(self.weight_alto)
        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches[:self.num_candidates]

        # Tenor #################################################################
        inputs_tenor = torch.cat([
            inputs_alto.repeat(self.num_candidates, 1),
            one_hot(results[:, 1].long(), part_size).float()
        ], dim=1)
        outputs_tenor = self.tenor(inputs_tenor)

        log_probabilities = torch.log(torch.softmax(outputs_tenor, dim=1))
        log_probabilities += math.log(self.weight_tenor)
        log_probabilities = log_probabilities.t() + results[:, 0]
        log_probabilities, pitches_indicies = torch.sort(log_probabilities.t().contiguous().view(1, -1).squeeze(), dim=0, descending=True)

        pitches_tenor = pitches_indicies % part_size
        history_indices = pitches_indicies // part_size
        pitches_alto = results[:, 1][history_indices]

        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches_alto[:self.num_candidates]
        results[:, 2] = pitches_tenor[:self.num_candidates]

        return results
