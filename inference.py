from argparse import ArgumentParser
import utils

import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

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
        bass,
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

    for step in tqdm(range(1, length), unit='time steps'):
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
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[
                       :num_candidates]
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

    return score

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

    for step in tqdm(range(1, length), unit='time steps'):
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
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[
                       :num_candidates]
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

    return bass_predicted


def compose_score(checkpoint_path, soprano_path):
    torch.set_grad_enabled(False)
    bass = predict_bass(
        soprano_path=soprano_path,
        checkpoint_path=checkpoint_path,
        num_candidates=1
    ).clone().detach().float()
    score = predict_middle_parts(
        bass,
        soprano_path=soprano_path,
        checkpoint_path=checkpoint_path,
        num_candidates=1
    )
    return score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="path to checkpoint .pt file", metavar="PATH")
    parser.add_argument("-m", "--musicxml", help="path to musicxml file")
    args = parser.parse_args()

    # python inference.py -c checkpoints/2021-04-06_18-18-01/2021-04-06_18-18-01_epoch=0142.pt -m data/musicxml/001_soprano.xml
    score = compose_score(args.checkpoint, args.musicxml)
    score.show('musicxml')
