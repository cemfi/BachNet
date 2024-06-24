import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

import utils
from data import pitch_sizes_parts, generate_data_inference, indices_parts
from model import BachNetInferenceContinuo, BachNetInferenceMiddleParts


# from music21 import environment
# environment.set('musicxmlPath', 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe')


def predict_middle_parts(bass, soprano_path, checkpoint_path, num_candidates=1):
    bass_for_playback = bass

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state_middle_parts']

    model = BachNetInferenceMiddleParts(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
        num_candidates=num_candidates
    )
    model.load_state_dict(state)
    model.set_part_weights(
        # loss_bass=checkpoint['loss_bass'],
        loss_alto=checkpoint['loss_alto'],
        loss_tenor=checkpoint['loss_tenor']
    )
    model.eval()

    sample = generate_data_inference(
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
    bass = torch.cat([
        torch.zeros((config.context_radius, bass.shape[1])),
        bass.float(),
        torch.zeros((config.context_radius, bass.shape[1]))
    ], dim=0)
    predictions = model({
        'soprano': inputs['soprano'][:2 * config.context_radius + 1],
        'alto': torch.zeros((config.context_radius, pitch_sizes_parts['alto'] + len(indices_parts))),
        'tenor': torch.zeros((config.context_radius, pitch_sizes_parts['tenor'] + len(indices_parts))),
        'bass_with_context': bass[:2 * config.context_radius + 1],
        'extra': inputs['extra'][:2 * config.context_radius + 1]
    })

    probabilities = [torch.max(predictions[:, 0]).item()]

    acc_probabilities = predictions[:, 0]
    history_pitches = [predictions[:, 1:]]

    for step in range(1, length):
        candidates = []
        padding_size = max(0, config.context_radius - len(history_pitches))
        history_size = min(config.context_radius, len(history_pitches))
        cur_history = torch.stack(history_pitches[-history_size:], dim=0).long()

        for candidate_idx in range(num_candidates):
            predictions = model({
                'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
                'bass_with_context': bass[step:step + 2 * config.context_radius + 1],
                'alto': torch.cat([
                    torch.zeros((padding_size, pitch_sizes_parts['alto'] + len(indices_parts))),
                    one_hot(cur_history[:, candidate_idx, 0], pitch_sizes_parts['alto'] + len(indices_parts)).float()
                ], dim=0),
                'tenor': torch.cat([
                    torch.zeros((padding_size, pitch_sizes_parts['tenor'] + len(indices_parts))),
                    one_hot(cur_history[:, candidate_idx, 1], pitch_sizes_parts['tenor'] + len(indices_parts)).float()
                ], dim=0),
                'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
            })
            candidates.append(predictions)

        candidates = torch.stack(candidates, dim=0)
        # 0: Index for chord in history_pitches[-1]
        # 1: Candidate idx
        # 2: Probability Pitches

        # Add log probabilities of candidates to current probabilities
        candidates[:, :, 0] = (acc_probabilities + candidates[:, :, 0].t()).t()

        candidate_indices = torch.argsort(candidates.view(-1, 4)[:, 0], dim=0, descending=True)
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[:num_candidates]
        # [[last_chord_idx, new_chord_idx]]

        history_pitches.append(torch.empty_like(history_pitches[-1]))
        for i in range(num_candidates):
            for k in range(len(history_pitches) - 2, -1, -1):
                history_pitches[k][i] = history_pitches[k][best_indices[i, 0]]
            history_pitches[-1][i] = candidates[best_indices[i, 0], best_indices[i, 1], 1:]
            acc_probabilities[i] = candidates[best_indices[i, 0], best_indices[i, 1], 0]

        probabilities.append(torch.max(acc_probabilities, dim=0)[0].item())

    winner = torch.stack(history_pitches, dim=1)[torch.argmax(acc_probabilities)].long().t()

    score = utils.tensors_to_stream({
        'soprano': soprano,
        'extra': extra,
        'bass': bass_for_playback,
        'alto': one_hot(winner[0], pitch_sizes_parts['alto'] + len(indices_parts)),
        'tenor': one_hot(winner[1], pitch_sizes_parts['tenor'] + len(indices_parts)),
    }, config, metadata)

    # for e in zip(winner.t(), probabilities):
    #     print(e)

    return score, probabilities[-1]

    # score.write('musicxml', f'beam_{num_candidates}.musicxml')
    # score.show('musicxml')


def predict_bass(soprano_path, checkpoint_path, num_candidates=1):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state_continuo']

    model = BachNetInferenceContinuo(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
        num_candidates=num_candidates,
    )
    model.load_state_dict(state)
    model.eval()

    sample = generate_data_inference(
        time_grid=config.time_grid,
        soprano_path=soprano_path
    )

    inputs = sample['data']
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
        'bass': torch.zeros((config.context_radius, pitch_sizes_parts['bass'] + len(indices_parts))),
        'extra': inputs['extra'][:2 * config.context_radius + 1]
    })
    probabilities = [torch.max(predictions[:, 0]).item()]
    acc_probabilities = predictions[:, 0]
    history_pitches = [predictions[:, 1:]]

    for step in range(1, length):
        candidates = []
        history_size = min(config.context_radius, len(history_pitches))
        padding_size = config.context_radius - history_size

        cur_history = torch.stack(history_pitches[-history_size:], dim=0).long()

        for candidate_idx in range(num_candidates):
            predictions = model({
                'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
                'bass': torch.cat([
                    torch.zeros((padding_size, pitch_sizes_parts['bass'] + len(indices_parts))),
                    one_hot(cur_history[:, candidate_idx, 0], pitch_sizes_parts['bass'] + len(indices_parts)).float()
                ], dim=0),
                'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
            })
            candidates.append(predictions)

        candidates = torch.stack(candidates, dim=0)
        # 0: Index for chord in history_pitches[-1]
        # 1: Candidate idx
        # 2: Probability Pitches

        # Add log probabilities of candidates to current probabilities
        candidates[:, :, 0] = (acc_probabilities + candidates[:, :, 0].t()).t()

        candidate_indices = torch.argsort(candidates.view(-1, 4)[:, 0], dim=0, descending=True)
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[:num_candidates]
        # [[last_chord_idx, new_chord_idx]]

        history_pitches.append(torch.empty_like(history_pitches[-1]))
        for i in range(num_candidates):
            for k in range(len(history_pitches) - 2, -1, -1):
                history_pitches[k][i] = history_pitches[k][best_indices[i, 0]]
            history_pitches[-1][i] = candidates[best_indices[i, 0], best_indices[i, 1], 1:]
            acc_probabilities[i] = candidates[best_indices[i, 0], best_indices[i, 1], 0]

        probabilities.append(torch.max(acc_probabilities, dim=0)[0].item())

    winner = torch.stack(history_pitches, dim=1)[torch.argmax(acc_probabilities)].long().t()

    # for e in zip(winner.t(), probabilities):
    #     print(e)
    bass_predicted = one_hot(winner[0], pitch_sizes_parts['bass'] + len(indices_parts))

    return bass_predicted, probabilities[-1]


if __name__ == '__main__':
    import os
    from glob import glob
    import re

    # checkpoint_paths = sorted(glob('./checkpoints/2019-06-17_17-10-47 batch_size=8192 hidden_size=650 context_radius=32 time_grid=0.25 lr=0.0005 lr_gamma=0.98 lr_step_size=20 split=0.05/*.pt'))
    checkpoint_paths = ['./checkpoints/2019-06-17_17-10-47 batch_size=8192 hidden_size=650 context_radius=32 time_grid=0.25 lr=0.0005 lr_gamma=0.98 lr_step_size=20 split=0.05/2019-06-17_17-10-47 batch_size=8192 hidden_size=650 context_radius=32 time_grid=0.25 lr=0.0005 lr_gamma=0.98 lr_step_size=20 split=0.05 0580.pt']
    expr = re.compile(r'hidden_size=(\d+)')

    table = []

    bass_bs = {
        '001': 1,
        '003': 6,
        '030': 2,
        '035': 1,
        '037': 1,
        '071': 1,
        '110': 1,
        '112': 1,
        '131': 2,
        '166': 4,
        '203': 1,
        '213': 1,
        '215': 2,
        '271': 2,
        '335': 2,
        '349': 1,
        '367': 2
    }

    middle_bs = {
        '001': 1,
        '003': 1,
        '030': 1,
        '035': 1,
        '037': 2,
        '071': 1,
        '110': 2,
        '112': 1,
        '131': 24,
        '166': 21,
        '203': 20,
        '213': 1,
        '215': 1,
        '271': 6,
        '335': 1,
        '349': 6,
        '367': 8
    }

    for cp in checkpoint_paths:
        min_losses = {}
        config = os.path.split(os.path.dirname(cp))[-1]
        hidden_size = expr.findall(config)[0]
        dirname = os.path.join('.', 'all_test_pieces_beam_search_final')
        os.makedirs(dirname, exist_ok=True)
        epoch = cp[-7:-3]

        for n in tqdm(['001', '003', '030', '035', '037', '071', '110', '112', '131', '166', '203', '213', '215', '271', '335', '349', '367']):
            os.makedirs(os.path.join(dirname, n), exist_ok=True)
            soprano_path = f'./data/musicxml/{n}_soprano.musicxml'

            bass, bass_loss = predict_bass(
                soprano_path=soprano_path,
                checkpoint_path=cp,
                num_candidates=bass_bs[n]
            )

            score, middle_loss = predict_middle_parts(
                bass,
                soprano_path=soprano_path,
                checkpoint_path=cp,
                num_candidates=middle_bs[n]
            )

            target_path_musicxml = os.path.join(dirname, n, f'no={n} middle={middle_loss}@{middle_bs[n]} bass={bass_loss}@{bass_bs[n]}.musicxml')
            score.write('musicxml', target_path_musicxml)
