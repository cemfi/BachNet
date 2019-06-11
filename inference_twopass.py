import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

import data
import utils
from model import BachNetInferenceContinuo, BachNetInferenceMiddleParts


# from music21 import environment
# environment.set('musicxmlPath', 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe')


def predict_middleparts(bass, soprano_path, checkpoint_path, num_candidates=1):
    # Disable autograd to save a huge amount of memory
    torch.set_grad_enabled(False)

    bass_for_playback = bass

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state_middleparts']

    model = BachNetInferenceMiddleParts(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
        num_candidates=num_candidates
    )
    model.load_state_dict(state)
    model.set_part_weights(
        #loss_bass=checkpoint['loss_bass'],
        loss_alto=checkpoint['loss_alto'],
        loss_tenor=checkpoint['loss_tenor']
    )
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
    bass = torch.cat([
            torch.zeros((config.context_radius, bass.shape[1])),
            bass,
            torch.zeros((config.context_radius, bass.shape[1]))
        ], dim=0)
    predictions = model({
        'soprano': inputs['soprano'][:2 * config.context_radius + 1],
        'alto': torch.zeros((config.context_radius, data.part_size)),
        'tenor': torch.zeros((config.context_radius, data.part_size)),
        'bass_withcontext': bass[:2 * config.context_radius + 1],
        'extra': inputs['extra'][:2 * config.context_radius + 1]
    })
    probabilities = [torch.max(predictions[:, 0]).item()]
    cur_probabilities = predictions[:, 0]
    history = [predictions[:, 1:]]

    for step in tqdm(range(1, length), unit='time steps'):
        candidates = []
        padding_size = max(0, config.context_radius - len(history))
        history_size = min(config.context_radius, len(history))
        cur_history = torch.stack(history[-history_size:], dim=0).long()

        for candidate_idx in range(num_candidates):
            predictions = model({
                'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
                'bass_withcontext': bass[step:step + 2 * config.context_radius + 1],
                'alto': torch.cat([
                    torch.zeros((padding_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 0], data.part_size).float()
                ], dim=0),
                'tenor': torch.cat([
                    torch.zeros((padding_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 1], data.part_size).float()
                ], dim=0),
                'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
            })
            candidates.append(predictions)

        candidates = torch.stack(candidates, dim=0)
        # 0: Index for chord in history[-1]
        # 1: Candidate idx
        # 2: Probability Pitches

        # Add log probabilities of candidates to current probabilities
        candidates[:, :, 0] = cur_probabilities.t() + candidates[:, :, 0]

        candidate_indices = torch.argsort(candidates.view(-1, 4)[:, 0], dim=0, descending=True)
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[:num_candidates]
        # [[last_chord_idx, new_chord_idx]]

        history.append(torch.empty_like(history[-1]))
        for i in range(num_candidates):
            history[-2][i] = history[-2][best_indices[i, 0]]
            history[-1][i] = candidates[best_indices[i, 0], best_indices[i, 1], 1:]
            cur_probabilities[i] = candidates[best_indices[i, 0], best_indices[i, 1], 0]

        probabilities.append(torch.max(cur_probabilities, dim=0)[0].item())

    winner = torch.stack(history, dim=1)[torch.argmax(cur_probabilities)].long().t()

    score = utils.tensors_to_stream({
        'soprano': soprano,
        'extra': extra,
        'bass': bass_for_playback,
        'alto': one_hot(winner[0], data.part_size),
        'tenor': one_hot(winner[1], data.part_size),
    }, config, metadata)

    for e in zip(winner.t(), probabilities):
        print(e)

    score.write('musicxml', f'beam_{num_candidates}.musicxml')
    score.show('musicxml')


def predict_bass(soprano_path, checkpoint_path, num_candidates=1):
    # Disable autograd to save a huge amount of memory
    torch.set_grad_enabled(False)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state_continuo']

    model = BachNetInferenceContinuo(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
        num_candidates=num_candidates,
        full_context_number=1,
        half_context_number=1
    )
    model.load_state_dict(state)
    model.set_part_weights(
        loss_bass=checkpoint['loss_bass'],
        #loss_alto=checkpoint['loss_alto'],
        #loss_tenor=checkpoint['loss_tenor']
    )
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
        #'alto': torch.zeros((config.context_radius, data.part_size)),
        #'tenor': torch.zeros((config.context_radius, data.part_size)),
        'bass': torch.zeros((config.context_radius, data.part_size)),
        'extra': inputs['extra'][:2 * config.context_radius + 1]
    })
    probabilities = [torch.max(predictions[:, 0]).item()]
    cur_probabilities = predictions[:, 0]
    history = [predictions[:, 1:]]

    for step in tqdm(range(1, length), unit='time steps'):
        candidates = []
        padding_size = max(0, config.context_radius - len(history))
        history_size = min(config.context_radius, len(history))
        cur_history = torch.stack(history[-history_size:], dim=0).long()

        for candidate_idx in range(num_candidates):
            predictions = model({
                'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
                'bass': torch.cat([
                    torch.zeros((padding_size, data.part_size)),
                    one_hot(cur_history[:, candidate_idx, 0], data.part_size).float()
                ], dim=0),
                #'alto': torch.cat([
                #    torch.zeros((padding_size, data.part_size)),
                #    one_hot(cur_history[:, candidate_idx, 1], data.part_size).float()
                #], dim=0),
                #'tenor': torch.cat([
                #    torch.zeros((padding_size, data.part_size)),
                #    one_hot(cur_history[:, candidate_idx, 2], data.part_size).float()
                #], dim=0),
                'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
            })
            candidates.append(predictions)

        candidates = torch.stack(candidates, dim=0)
        # 0: Index for chord in history[-1]
        # 1: Candidate idx
        # 2: Probability Pitches

        # Add log probabilities of candidates to current probabilities
        candidates[:, :, 0] = cur_probabilities.t() + candidates[:, :, 0]

        candidate_indices = torch.argsort(candidates.view(-1, 4)[:, 0], dim=0, descending=True)
        best_indices = torch.stack([candidate_indices // num_candidates, candidate_indices % num_candidates], dim=1)[:num_candidates]
        # [[last_chord_idx, new_chord_idx]]

        history.append(torch.empty_like(history[-1]))
        for i in range(num_candidates):
            history[-2][i] = history[-2][best_indices[i, 0]]
            history[-1][i] = candidates[best_indices[i, 0], best_indices[i, 1], 1:]
            cur_probabilities[i] = candidates[best_indices[i, 0], best_indices[i, 1], 0]

        probabilities.append(torch.max(cur_probabilities, dim=0)[0].item())

    winner = torch.stack(history, dim=1)[torch.argmax(cur_probabilities)].long().t()

    for e in zip(winner.t(), probabilities):
        print(e)
    bass_predicted = one_hot(winner[0], data.part_size)

    for e in zip(winner.t(), probabilities):
        print(e)

    return bass_predicted


if __name__ == '__main__':
    import os
    from glob import glob

    latest_checkpoint = sorted(glob('./checkpoints/*.pt'))[-1]
    print(os.path.split(latest_checkpoint)[-1])

    bass = predict_bass(
        soprano_path='./data/musicxml/029_soprano.musicxml',
        # checkpoint_path='./checkpoints/2019-06-09_19-31-32 batch_size=8192 hidden_size=800 context_radius=32 time_grid=0.25 lr=0.001 lr_gamma=0.98 lr_step_size=10 split=0.05 0020.pt',
        checkpoint_path=latest_checkpoint,
        num_candidates=1
    ).clone().float()
    predict_middleparts(
        bass,
        soprano_path='./data/musicxml/029_soprano.musicxml',
        # checkpoint_path='./checkpoints/2019-06-09_19-31-32 batch_size=8192 hidden_size=800 context_radius=32 time_grid=0.25 lr=0.001 lr_gamma=0.98 lr_step_size=10 split=0.05 0020.pt',
        checkpoint_path=latest_checkpoint,
        num_candidates=1
    )
