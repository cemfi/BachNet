import math

import torch.nn
from torch.nn.functional import one_hot

from data import indices_extra, pitch_sizes_parts, indices_parts


class BachNetTrainingContinuo(torch.nn.Module):
    def __init__(self, hidden_size, context_radius, dropout=0.5):
        super(BachNetTrainingContinuo, self).__init__()

        self.bass = torch.nn.Sequential(
            torch.nn.Linear(
                (2 * context_radius + 1) * (pitch_sizes_parts['soprano'] + len(indices_parts)) +
                (2 * context_radius + 1) * len(indices_extra) +
                context_radius * (pitch_sizes_parts['bass'] + len(indices_parts)),

                hidden_size
            ),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, pitch_sizes_parts['bass'] + len(indices_parts)),
        )

    def forward(self, inputs):
        batch_size = inputs['soprano'].shape[0]
        inputs_bass = torch.cat([inputs[k].view(batch_size, -1) for k in ['soprano', 'bass', 'extra']], dim=1)

        outputs_bass = self.bass(inputs_bass)

        return {
            'bass': outputs_bass
        }


class BachNetTrainingMiddleParts(torch.nn.Module):
    def __init__(self, hidden_size, context_radius, dropout=0.5):
        super(BachNetTrainingMiddleParts, self).__init__()

        self.alto = torch.nn.Sequential(
            torch.nn.Linear(
                (2 * context_radius + 1) * (pitch_sizes_parts['soprano'] + len(indices_parts)) +
                (2 * context_radius + 1) * (pitch_sizes_parts['bass'] + len(indices_parts)) +
                (2 * context_radius + 1) * len(indices_extra) +
                context_radius * (pitch_sizes_parts['alto'] + len(indices_parts)) +
                context_radius * (pitch_sizes_parts['tenor'] + len(indices_parts)),

                hidden_size
            ),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, pitch_sizes_parts['alto'] + len(indices_parts)),
        )

        self.tenor = torch.nn.Sequential(
            torch.nn.Linear(
                (2 * context_radius + 1) * (pitch_sizes_parts['soprano'] + len(indices_parts)) +
                (2 * context_radius + 1) * (pitch_sizes_parts['bass'] + len(indices_parts)) +
                (2 * context_radius + 1) * len(indices_extra) +
                context_radius * (pitch_sizes_parts['alto'] + len(indices_parts)) +
                context_radius * (pitch_sizes_parts['tenor'] + len(indices_parts)) +
                (pitch_sizes_parts['alto'] + len(indices_parts)),

                hidden_size
            ),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, pitch_sizes_parts['tenor'] + len(indices_parts)),
        )

    def forward(self, inputs):
        batch_size = inputs['soprano'].shape[0]
        inputs_alto = torch.cat(
            [inputs[k].view(batch_size, -1) for k in ['soprano', 'alto', 'tenor', 'bass_with_context', 'extra']], dim=1)

        outputs_alto = self.alto(inputs_alto)
        prediction_alto = one_hot(torch.max(outputs_alto, dim=1)[1],
                                  pitch_sizes_parts['alto'] + len(indices_parts)).float()

        inputs_tenor = torch.cat([inputs_alto, prediction_alto], dim=1)
        outputs_tenor = self.tenor(inputs_tenor)

        return {
            'alto': outputs_alto,
            'tenor': outputs_tenor
        }


class BachNetInferenceContinuo(BachNetTrainingContinuo):
    def __init__(self, num_candidates, *args, **kwargs):
        super(BachNetInferenceContinuo, self).__init__(*args, **kwargs)
        self.num_candidates = num_candidates

    def forward(self, inputs):
        num_parts = 3
        results = torch.zeros(
            (self.num_candidates, 1 + num_parts))  # [[Candidate index], [[ProbAcc, PitchB, PitchA, PitchT]]

        inputs_bass = torch.cat([
            inputs[k].view(1, -1) for k in ['soprano', 'bass', 'extra']
        ], dim=1).squeeze()

        outputs_bass = self.bass(inputs_bass)

        log_probabilities, pitches = torch.sort(torch.log(torch.softmax(outputs_bass, dim=0)), dim=0, descending=True)
        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches[:self.num_candidates]

        return results


class BachNetInferenceMiddleParts(BachNetTrainingMiddleParts):
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
        results = torch.zeros(
            (self.num_candidates, 1 + num_parts))  # [[Candidate index], [[ProbAcc, PitchB, PitchA, PitchT]]

        # Alto #################################################################
        inputs_alto = torch.cat(
            [inputs[k].view(1, -1) for k in ['soprano', 'alto', 'tenor', 'bass_with_context', 'extra']],
            dim=1).squeeze()  # !!! SQUEEZED !!!
        outputs_alto = self.alto(inputs_alto)

        log_probabilities, pitches = torch.sort(torch.log(torch.softmax(outputs_alto, dim=0)), dim=0, descending=True)
        log_probabilities += math.log(self.weight_alto)
        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches[:self.num_candidates]

        # Tenor #################################################################
        inputs_tenor = torch.cat([
            inputs_alto.repeat(self.num_candidates, 1),
            one_hot(results[:, 1].long(), pitch_sizes_parts['alto'] + len(indices_parts)).float()
        ], dim=1)
        outputs_tenor = self.tenor(inputs_tenor)

        log_probabilities = torch.log(torch.softmax(outputs_tenor, dim=1))
        log_probabilities += math.log(self.weight_tenor)
        log_probabilities = log_probabilities.t() + results[:, 0]
        log_probabilities, pitches_indices = torch.sort(log_probabilities.t().contiguous().view(1, -1).squeeze(), dim=0,
                                                        descending=True)

        pitches_tenor = pitches_indices % (pitch_sizes_parts['tenor'] + len(indices_parts))
        history_indices = pitches_indices // (pitch_sizes_parts['tenor'] + len(indices_parts))
        pitches_alto = results[:, 1][history_indices]

        results[:, 0] = log_probabilities[:self.num_candidates]
        results[:, 1] = pitches_alto[:self.num_candidates]
        results[:, 2] = pitches_tenor[:self.num_candidates]

        return results
