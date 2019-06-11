import math
import os
import random
import shutil
from glob import glob

import torch
from music21 import converter
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
    'num_sharps': 1,
    'time_numerator': 2,
    'time_denominator': 3,
    'time_pos': 4
}

pitch_size = 60
part_size = pitch_size + len(indices_parts)


class ChoralesDataset(Dataset):
    def __init__(self, root_dir, context_radius=32, transpositions=[0]):
        self.root_dir = root_dir
        self.context_radius = context_radius

        # Make empty intros for each part
        self.data = {
            'soprano': [torch.zeros((context_radius, part_size))],
            'tenor': [torch.zeros((context_radius, part_size))],
            'alto': [torch.zeros((context_radius, part_size))],
            'bass': [torch.zeros((context_radius, part_size))],
            'extra': [torch.zeros((context_radius, len(indices_extra)))]
        }

        # Concat all pieces into large tensors for each part
        for file_path in sorted(glob(os.path.join(self.root_dir, '*.pt'))):
            data = torch.load(file_path)['data']
            for part_name, part_data in data.items():
                part_transposed = []
                for t in transpositions:
                    if part_name == 'extra':
                        sharps_offset = (t * 7) % 12
                        part_cloned = part_data.clone()
                        part_cloned[:, indices_extra['num_sharps']] = (part_cloned[:, indices_extra['num_sharps']] + sharps_offset) % 12
                        part_transposed.append(part_cloned)
                    else:
                        part_transposed.append(torch.cat([
                            part_data[:, :len(indices_parts)],
                            part_data[:, len(indices_parts):].roll(t, dims=1)
                        ], dim=1))
                    part_transposed.append(torch.zeros((context_radius, part_data.shape[1])))
                self.data[part_name].append(torch.cat(part_transposed, dim=0))

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
                   'bass_withcontext': self.data['bass'][idx:idx + 2 * self.context_radius + 1],
                   'bass': self.data['bass'][idx:idx + self.context_radius],
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
        'soprano': torch.zeros((length, part_size))
    }

    # Iterate through all musical elements in current voice stream
    for element in stream.flat:
        offset = int(element.offset / time_grid)

        if type(element) == Note:
            # Skip grace notes
            if element.duration.quarterLength == 0:
                continue

            pitch = element.pitch.midi - pitch_size // 2 + len(indices_parts)
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
            data['extra'][offset:, indices_extra['time_numerator']] = element.numerator
            data['extra'][offset:, indices_extra['time_denominator']] = element.denominator

        if type(element) == KeySignature or type(element) == Key:
            data['extra'][offset:, indices_extra['num_sharps']] = element.sharps

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
        'metadata': stream.metadata
    }


def _generate_data_training(time_grid, root_dir, overwrite, split):
    target_dir = os.path.join(root_dir, f'time_grid={time_grid} split={split}')

    if os.path.exists(target_dir) and not overwrite:
        return target_dir

    if overwrite:
        shutil.rmtree(root_dir)

    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    musicxml_dir = os.path.join(root_dir, 'musicxml')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(musicxml_dir, exist_ok=True)

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

        length = math.ceil(streams['soprano'].highestTime / time_grid)
        data = {
            'extra': torch.zeros((length, len(indices_extra)))
        }
        for part_name, part in streams.items():
            part = part.flat
            # Init empty tensor for current voice
            data[part_name] = torch.zeros((length, part_size))

            # Iterate through all musical elements in current voice stream
            for element in part:
                offset = int(element.offset / time_grid)

                if type(element) == Note:
                    # Skip grace notes
                    if element.duration.quarterLength == 0:
                        continue

                    pitch = element.pitch.midi - pitch_size // 2 + len(indices_parts)
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
                        data['extra'][offset:, indices_extra['time_numerator']] = element.numerator
                        data['extra'][offset:, indices_extra['time_denominator']] = element.denominator

                    if type(element) == KeySignature or type(element) == Key:
                        data['extra'][offset:, indices_extra['num_sharps']] = element.sharps

        measure_offsets = [o / time_grid for o in streams['soprano'].measureOffsetMap().keys()]
        cur_offset = streams['soprano'].flat.notesAndRests[0].beat
        data['extra'][0, indices_extra['time_pos']] = cur_offset
        for offset in range(1, length):
            if offset in measure_offsets:
                cur_offset = 1
            else:
                cur_offset += time_grid
            data['extra'][offset, indices_extra['time_pos']] = cur_offset

        target_file_path = os.path.join(target_dir, f'{str(chorale.metadata.number).zfill(3)}.pt')
        torch.save({
            'data': data,
            'title': chorale.metadata.title
        }, target_file_path)

        chorale.write('musicxml', os.path.join(musicxml_dir, f'{str(chorale.metadata.number).zfill(3)}_full.musicxml'))
        chorale['Soprano'].write('musicxml', os.path.join(musicxml_dir, f'{str(chorale.metadata.number).zfill(3)}_soprano.musicxml'))

    # Move files to train / test directories
    file_paths = glob(os.path.join(target_dir, '*.pt'))
    random.shuffle(file_paths)  # Shuffle in place
    split_idx = int(len(file_paths) * split)
    for file_path in file_paths[split_idx:]:
        shutil.move(file_path, train_dir)
    for file_path in file_paths[:split_idx]:
        shutil.move(file_path, test_dir)

    return target_dir


def _make_data_loaders(root_dir, batch_size, num_workers, context_radius, transpositions):
    # Training data loader: random sampling
    train_dataset = ChoralesDataset(
        root_dir=os.path.join(root_dir, 'train'),
        context_radius=context_radius,
        transpositions=transpositions
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
        context_radius=context_radius,
        transpositions=[0]
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
        'test': test_data_loader,
        'input_size': part_size * 4 + len(indices_extra),
        'output_sizes': part_size
    }


def get_data_loaders(time_grid=0.25, root_dir=None, overwrite=False, split=0.15, batch_size=1, num_workers=1, context_radius=32, transpositions=[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]):
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
        transpositions=transpositions,
        batch_size=batch_size,
        num_workers=num_workers,
        context_radius=context_radius
    )

    return data_loaders


if __name__ == '__main__':
    get_data_loaders(overwrite=True)
