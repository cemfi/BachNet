from copy import deepcopy

import math
import os
import random
import shutil
from glob import glob

import torch
from music21 import converter
from music21.analysis.discrete import Ambitus
from music21.corpus import chorales
from music21.expressions import Fermata
from music21.key import KeySignature, Key
from music21.meter import TimeSignature
from music21.note import Note, Rest
from torch.utils.data import Dataset, RandomSampler, BatchSampler, SequentialSampler, DataLoader
from tqdm import tqdm

indices_parts = {
    'is_continued': 0,
    'is_rest': 1
}

indices_extra = {
    'has_fermata': 0,
    'has_time_signature_3/4': 1,
    'has_time_signature_4/4': 2,
    'has_time_signature_3/2': 3,
    'time_pos': 4,
    'pitch_offset': 5
}

sharps_to_offset = {
    -4: -4,
    -3: 3,
    -2: -2,
    -1: 5,
    0: 0,
    1: -5,
    2: 2,
    3: -3,
    4: 4
}

offset_to_sharps = {
    -5: 1,
    -4: -4,
    -3: 3,
    -2: -2,
    0: 0,
    2: 2,
    3: -3,
    4: 4,
    5: -1
}

min_pitches = {
    'bass': 36,
    'tenor': 48,
    'alto': 53,
    'soprano': 57
}

max_pitches = {
    'bass': 64,
    'tenor': 69,
    'alto': 74,
    'soprano': 81
}

pitch_sizes_parts = {}
for part_name in min_pitches.keys():
    pitch_sizes_parts[part_name] = max_pitches[part_name] - min_pitches[part_name] + 1

ambitus = Ambitus()


class ChoralesDataset(Dataset):
    def __init__(self, root_dir, context_radius=32):
        self.root_dir = root_dir
        self.context_radius = context_radius

        # Make empty intros for each part
        self.data = {
            'soprano': [torch.zeros((context_radius, pitch_sizes_parts['soprano'] + len(indices_parts)))],
            'tenor': [torch.zeros((context_radius, pitch_sizes_parts['tenor'] + len(indices_parts)))],
            'alto': [torch.zeros((context_radius, pitch_sizes_parts['alto'] + len(indices_parts)))],
            'bass': [torch.zeros((context_radius, pitch_sizes_parts['bass'] + len(indices_parts)))],
            'extra': [torch.zeros((context_radius, len(indices_extra)))]
        }

        # Concat all pieces into large tensors for each part
        for file_path in sorted(glob(os.path.join(self.root_dir, '*.pt'))):
            data = torch.load(file_path)['data']
            for part_name, part_data in data.items():
                self.data[part_name].append(torch.cat([part_data, torch.zeros((context_radius, part_data.shape[1]))], dim=0))

        for part_name, part_data in self.data.items():
            self.data[part_name] = torch.cat(part_data, dim=0)

    def __len__(self):
        return self.data['soprano'].shape[0] - 2 * self.context_radius

    def __getitem__(self, idx):
        # Return windowed parts from dataset for training and correct "pitch classes" as targets
        return {
                   'soprano': self.data['soprano'][idx:idx + 2 * self.context_radius + 1],
                   'alto': self.data['alto'][idx:idx + self.context_radius],
                   'tenor': self.data['tenor'][idx:idx + self.context_radius],
                   'bass': self.data['bass'][idx:idx + self.context_radius],

                   'bass_with_context': self.data['bass'][idx:idx + 2 * self.context_radius + 1],
                   'extra': self.data['extra'][idx:idx + 2 * self.context_radius + 1]
               }, {
                   'alto': torch.argmax(self.data['alto'][idx + self.context_radius]),
                   'tenor': torch.argmax(self.data['tenor'][idx + self.context_radius]),
                   'bass': torch.argmax(self.data['bass'][idx + self.context_radius])
               }


def generate_data_inference(time_grid, soprano_path):
    stream = converter.parse(soprano_path)

    length = math.ceil(stream.highestTime / time_grid)
    data = {
        'extra': torch.zeros((length, len(indices_extra))),
        'soprano': torch.zeros((length, pitch_sizes_parts['soprano'] + len(indices_parts)))
    }

    # Transpose chorale to C/Am
    keys = list(stream.flat.getElementsByClass(Key))
    if len(keys) > 0:
        transposition = sharps_to_offset[keys[0].sharps]
        stream.transpose(-transposition, inPlace=True)
    else:
        transposition = 0

    # Iterate through all musical elements in current voice stream
    for element in stream.flat:
        offset = int(element.offset / time_grid)

        if type(element) == Note:
            # Skip grace notes
            if element.duration.quarterLength == 0:
                continue

            pitch = element.pitch.midi - min_pitches['soprano'] + len(indices_parts)
            duration = int(element.duration.quarterLength / time_grid)

            # Store pitch and ties
            data['soprano'][offset, pitch] = 1
            data['soprano'][offset + 1:offset + duration, indices_parts['is_continued']] = 1

            # Fermata
            if any([type(e) == Fermata for e in element.expressions]):
                data['extra'][offset, indices_extra['has_fermata']] = 1

        if type(element) == Rest:
            duration = int(element.duration.quarterLength / time_grid)
            data['soprano'][offset, indices_parts['is_rest']] = 1
            data['soprano'][offset + 1:offset + duration, indices_parts['is_continued']] = 1

        if type(element) == TimeSignature:
            if element.ratioString == '3/4':
                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 1
                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
            elif element.ratioString == '4/4':
                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 1
                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
            elif element.ratioString == '3/2':
                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 1

    measure_offsets = [o / time_grid for o in stream.measureOffsetMap().keys()]
    cur_offset = stream.flat.notesAndRests[0].beat
    data['extra'][0, indices_extra['time_pos']] = cur_offset
    for offset in range(1, length):
        if offset in measure_offsets:
            cur_offset = 1
        else:
            cur_offset += time_grid
        data['extra'][offset, indices_extra['time_pos']] = cur_offset

    return {
        'data': data,
        'metadata': stream.metadata,
        'transposition': transposition
    }


def _generate_data_training(time_grid, root_dir, overwrite, split):
    target_dir = os.path.join(root_dir, f'time_grid={time_grid} split={split}')

    if os.path.exists(target_dir) and not overwrite:
        return target_dir

    if overwrite and os.path.exists(target_dir):
        shutil.rmtree(root_dir)

    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    musicxml_dir = os.path.join(root_dir, 'musicxml')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(musicxml_dir, exist_ok=True)

    chorale_numbers = []

    for chorale in tqdm(chorales.Iterator(returnType='stream'), unit='chorales', desc='Generating dataset'):
        # Skip chorales with more or less than 4 parts
        if len(chorale.parts) != 4:
            continue

        # Skip if parts do not contain correct choral voices
        try:
            streams = {
                'soprano': chorale['Soprano'],
                'alto': chorale['Alto'],
                'tenor': chorale['Tenor'],
                'bass': chorale['Bass']
            }
        except KeyError:
            continue

        # Save soprano in own file for inference
        chorale['Soprano'].write('musicxml', os.path.join(musicxml_dir, f'{str(chorale.metadata.number).zfill(3)}_soprano.musicxml'))
        chorale.write('musicxml', os.path.join(musicxml_dir, f'{str(chorale.metadata.number).zfill(3)}_full.musicxml'))

        # Get minimum and maximum transpositions
        transpositions_down = -float('inf')
        transpositions_up = float('inf')
        for part_name, part in streams.items():
            min_pitch, max_pitch = ambitus.getPitchSpan(part)
            transpositions_down = max(transpositions_down, min_pitches[part_name] - min_pitch.midi)
            transpositions_up = min(transpositions_up, max_pitches[part_name] - max_pitch.midi)

        length = math.ceil(streams['soprano'].highestTime / time_grid)
        for t in range(transpositions_down, transpositions_up + 1):
            data = {'extra': torch.zeros((length, len(indices_extra)))}
            # Note transposition offset
            data['extra'][:, indices_extra['pitch_offset']] = t
            for part_name, part in streams.items():
                part = deepcopy(part)
                part = part.flat.transpose(t)
                # Init empty tensor for current voice
                data[part_name] = torch.zeros((length, pitch_sizes_parts[part_name] + len(indices_parts)))

                # Iterate through all musical elements in current voice stream
                for element in part:
                    offset = int(element.offset / time_grid)

                    if type(element) == Note:
                        # Skip grace notes
                        if element.duration.quarterLength == 0:
                            continue

                        pitch = element.pitch.midi - min_pitches[part_name] + len(indices_parts)
                        duration = int(element.duration.quarterLength / time_grid)

                        # Store pitch and ties
                        data[part_name][offset, pitch] = 1
                        data[part_name][offset + 1:offset + duration, indices_parts['is_continued']] = 1

                        # Fermata (only used in soprano)
                        if part_name == 'soprano' and any([type(e) == Fermata for e in element.expressions]):
                            data['extra'][offset, indices_extra['has_fermata']] = 1

                    if type(element) == Rest:
                        duration = int(element.duration.quarterLength / time_grid)
                        data[part_name][offset, indices_parts['is_rest']] = 1
                        data[part_name][offset + 1:offset + duration, indices_parts['is_continued']] = 1

                    if part_name == 'soprano':
                        if type(element) == TimeSignature:
                            if element.ratioString == '3/4':
                                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 1
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
                            elif element.ratioString == '4/4':
                                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 1
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
                            elif element.ratioString == '3/2':
                                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 1

            measure_offsets = [o / time_grid for o in streams['soprano'].measureOffsetMap().keys()]
            cur_offset = streams['soprano'].flat.notesAndRests[0].beat
            data['extra'][0, indices_extra['time_pos']] = cur_offset
            for offset in range(1, length):
                if offset in measure_offsets:
                    cur_offset = 1
                else:
                    cur_offset += time_grid
                data['extra'][offset, indices_extra['time_pos']] = cur_offset

            signum = '+' if t >= 0 else '-'

            target_file_path = os.path.join(target_dir, f'{str(chorale.metadata.number).zfill(3)}{signum}{int(math.fabs(t))}.pt')
            torch.save({
                'data': data,
                'title': chorale.metadata.title
            }, target_file_path)

        chorale_numbers.append(str(chorale.metadata.number).zfill(3))

    # Move files to train / test directories
    random.shuffle(chorale_numbers)  # Shuffle in place
    split_idx = int(len(chorale_numbers) * split)

    for cn in chorale_numbers[split_idx:]:  # Train
        file_paths = glob(os.path.join(target_dir, f'*{cn}*.pt'))
        for file_path in file_paths:
            shutil.move(file_path, train_dir)

    for cn in chorale_numbers[:split_idx]:  # Test
        file_paths = glob(os.path.join(target_dir, f'*{cn}+0.pt'))
        for file_path in file_paths:
            shutil.move(file_path, test_dir)

    remove_paths = glob(os.path.join(target_dir, '*.pt'))
    for remove_path in remove_paths:
        os.remove(remove_path)

    return target_dir


def _make_data_loaders(root_dir, batch_size, num_workers, context_radius):
    # Training data loader: random sampling
    train_dataset = ChoralesDataset(
        root_dir=os.path.join(root_dir, 'train'),
        context_radius=context_radius
    )
    train_sampler = RandomSampler(train_dataset)
    train_batch_sampler = BatchSampler(
        train_sampler, batch_size, drop_last=False
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers
    )

    # Testing data loader: sequential sampling
    test_dataset = ChoralesDataset(
        root_dir=os.path.join(root_dir, 'test'),
        context_radius=context_radius
    )
    test_sampler = SequentialSampler(test_dataset)
    test_batch_sampler = BatchSampler(
        test_sampler, batch_size, drop_last=False
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=num_workers
    )

    return {
        'train': train_data_loader,
        'test': test_data_loader
    }


def get_data_loaders(time_grid=0.25, root_dir=None, overwrite=False, split=0.05, batch_size=1, num_workers=1, context_radius=32):
    if root_dir is None:
        root_dir = os.path.join('.', 'data')

    data_dir = _generate_data_training(
        time_grid=time_grid,
        root_dir=root_dir,
        overwrite=overwrite,
        split=split
    )

    data_loaders = _make_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        context_radius=context_radius
    )

    return data_loaders


if __name__ == '__main__':
    get_data_loaders(overwrite=True)
