import torch
from torch.nn.functional import one_hot

import data
import utils
from model import BachNetInferenceWithBeamSearch

from music21 import environment

environment.set('musicxmlPath', 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe')


def main(soprano_path, checkpoint_path, num_candidates):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state']

    model = BachNetInferenceWithBeamSearch(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
        num_candidates=num_candidates
    )
    model.load_state_dict(state)
    model.eval()

    sample = data.generate_data_inference(
        time_grid=config.time_grid,
        soprano_path=soprano_path
    )

    inputs = sample['data']
    metadata = sample['metadata']

    soprano = inputs['soprano'].clone()
    extra = inputs['extra'].clone()

    length = inputs['soprano'].shape[0]

    # Zero padding for input data
    for part in ['soprano', 'extra']:
        inputs[part] = torch.cat([
            torch.zeros((config.context_radius, inputs[part].shape[1])),
            inputs[part],
            torch.zeros((config.context_radius, inputs[part].shape[1]))
        ], dim=0)

    predictions = model({
        'soprano': inputs['soprano'][:2 * config.context_radius + 1],
        'alto': torch.zeros((config.context_radius, data.part_size)),
        'tenor': torch.zeros((config.context_radius, data.part_size)),
        'bass': torch.zeros((config.context_radius, data.part_size)),
        'extra': inputs['extra'][:2 * config.context_radius + 1]
    })
    cur_probabilities = predictions[:, 0]
    history = [predictions[:, 1:]]

    for step in range(1, length):
        candidates = []
        zeros_size = max(0, config.context_radius - len(history))
        history_size = min(config.context_radius, len(history))
        cur_history = torch.stack(history[-history_size:], dim=0).long()

        for candidate_idx in range(num_candidates):
            predictions = model({
                'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
                'bass': torch.cat([
                    torch.zeros((zeros_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 0], data.part_size).float()
                ], dim=0),
                'alto': torch.cat([
                    torch.zeros((zeros_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 1], data.part_size).float()
                ], dim=0),
                'tenor': torch.cat([
                    torch.zeros((zeros_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 2], data.part_size).float()
                ], dim=0),
                'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
            })
            candidates.append(predictions)

        print(candidates)

        candidates = torch.stack(candidates, dim=0)
        # 0: Index for chord in history[-1]
        # 1: Candidate idx
        # 2: Probability 1 Pitches

        print(candidates)

        # Multiply probabilities of candidates with current probabilities
        candidates[:, :, 0] = cur_probabilities.t() * candidates[:, :, 0]

        print(cur_probabilities)
        print(candidates)

        exit()

        probabilities, candidate_indices = torch.sort(candidates.view(-1, 4)[:, 0], dim=0, descending=True)
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[:num_candidates]
        # [[last_chord_idx, new_chord_idx]]



        history.append(torch.empty_like(history[-1]))
        for i in range(num_candidates):
            history[-2][i] = history[-2][best_indices[i, 0]]
            history[-1][i] = candidates[best_indices[i, 0], best_indices[i, 1], 1:]
            cur_probabilities[i] = candidates[best_indices[i, 0], best_indices[i, 1], 0]

    winner = torch.stack(history, dim=1)[torch.argmax(cur_probabilities)].long()

    # print(winner)
    print(torch.max(cur_probabilities).item())

    outputs = {
        'soprano': soprano,
        'extra': extra,
        'bass': one_hot(winner.t()[0], data.part_size),
        'alto': one_hot(winner.t()[1], data.part_size),
        'tenor': one_hot(winner.t()[2], data.part_size),
    }

    score = utils.tensors_to_stream(outputs, config, metadata)
    # score.show('musicxml')
    score.write('musicxml', 'output.musicxml')


if __name__ == '__main__':
    # from glob import glob
    # latest_checkpoint = sorted(glob('./checkpoints/**/*.pt'))[-1]

    main(
        soprano_path='./data/musicxml/230 Christ, der du bist der helle Tag_soprano.musicxml',
        checkpoint_path='./0070 batch_size=8192 hidden_size=500 context_radius=32 time_grid=0.25 lr=0.002.pt',
        num_candidates=3
    )