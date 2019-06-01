import math
import os
import random
import shutil
from glob import glob

import torch
from easydict import EasyDict
from music21.corpus import chorales
from music21.expressions import Fermata
from music21.key import KeySignature, Key
from music21.meter import TimeSignature
from music21.note import Note, Rest
from torch.utils.data import Dataset, RandomSampler, BatchSampler, SequentialSampler, DataLoader
from tqdm import tqdm

indices_parts = EasyDict({
    'is_continued': 0,
    'is_rest': 1
})

indices_extra = EasyDict({
    'has_fermata': 0,
    'num_sharps': 1,
    'time_numerator': 2,
    'time_denominator': 3,
    'time_pos': 4
})


class ChoralesDataset(Dataset):
    def __init__(self, root_dir, context_radius=32):
        self.root_dir = root_dir
        self.context_radius = context_radius

        # Make empty intros for each part
        self.data = EasyDict({
            'soprano': [torch.zeros((context_radius, 60 + len(indices_parts)))],
            'tenor': [torch.zeros((context_radius, 60 + len(indices_parts)))],
            'alto': [torch.zeros((context_radius, 60 + len(indices_parts)))],
            'bass': [torch.zeros((context_radius, 60 + len(indices_parts)))],
            'extra': [torch.zeros((context_radius, len(indices_extra)))]
        })

        # Concat all pieces into large tensors for each part
        for file_path in glob(os.path.join(self.root_dir, '*.pt')):
            data = torch.load(file_path)['data']
            for part_name, part_data in data.items():
                self.data[part_name].append(
                    torch.cat((part_data, torch.zeros((context_radius, part_data.shape[1]))), dim=0))
        for part_name, part_data in self.data.items():
            self.data[part_name] = torch.cat(self.data[part_name], dim=0).float()

    def __len__(self):
        return self.data.soprano.shape[0] - 2 * self.context_radius

    def __getitem__(self, idx):
        # Return windowed parts from dataset for training and one hot vectors as targets
        # "Future" of A+T+B is filled with zeros
        return {
                   'soprano': self.data.soprano[idx:idx + 2 * self.context_radius + 1],
                   'alto': torch.cat((self.data.alto[idx:idx + self.context_radius], torch.zeros((self.context_radius + 1, 60 + len(indices_parts))))),
                   'tenor': torch.cat((self.data.tenor[idx:idx + self.context_radius], torch.zeros((self.context_radius + 1, 60 + len(indices_parts))))),
                   'bass': torch.cat((self.data.bass[idx:idx + self.context_radius], torch.zeros((self.context_radius + 1, 60 + len(indices_parts))))),
                   'extra': self.data.extra[idx:idx + 2 * self.context_radius + 1]
               }, {
                   'alto': self.data.alto[idx + self.context_radius + 1].long(),
                   'tenor': self.data.tenor[idx + self.context_radius + 1].long(),
                   'bass': self.data.bass[idx + self.context_radius + 1].long()
               }


def _generate_data(time_grid, root_dir, overwrite, split):
    target_dir = os.path.join(root_dir, f'time_grid={time_grid} split={split}')

    if os.path.exists(target_dir) and not overwrite:
        return target_dir

    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for chorale in tqdm(chorales.Iterator(returnType='stream'), unit='chorales', desc='Generating dataset'):
        # Skip chorales with more or less than 4 parts
        if len(chorale.parts) != 4:
            continue

        # Skip if parts do not contain correct choral voices
        try:
            streams = {
                'soprano': chorale['Soprano'].flat,
                'alto': chorale['Alto'].flat,
                'tenor': chorale['Tenor'].flat,
                'bass': chorale['Bass'].flat
            }
        except KeyError:
            continue

        length = math.ceil(streams['soprano'].highestTime / time_grid)
        data = EasyDict({
            'extra': torch.zeros((length, len(indices_extra)))
        })
        for part_name, part in streams.items():
            # Init empty tensor for current voice
            data[part_name] = torch.zeros((length, 60 + len(indices_parts)))

            # Iterate through all musical elements in current voice stream
            for element in part:
                offset = int(element.offset / time_grid)

                if type(element) == Note:
                    # Skip grace notes
                    if element.duration.quarterLength == 0:
                        continue

                    pitch = element.pitch.midi - 30 + len(indices_parts)
                    duration = int(element.duration.quarterLength / time_grid)

                    # Store pitch and ties
                    data[part_name][offset, pitch] = 1
                    data[part_name][offset + 1:offset + duration, indices_parts.is_continued] = 1

                    # Fermata (only used in soprano)
                    if part_name == 'soprano' and any([type(e) == Fermata for e in element.expressions]):
                        data.extra[offset, indices_extra.has_fermata] = 1

                    # Save position ("beat") in measure
                    if part_name == 'soprano':
                        data.extra[offset, indices_extra.time_pos] = element.beat

                if type(element) == Rest:
                    duration = int(element.duration.quarterLength / time_grid)
                    data[part_name][offset, indices_parts.is_rest] = 1
                    data[part_name][offset + 1:offset + duration, indices_parts.is_continued] = 1

                # Additional information only relevant in soprano voice a.k.a. the model input
                if part_name == 'soprano':
                    if type(element) == TimeSignature:
                        data.extra[offset:, indices_extra.time_numerator] = element.numerator
                        data.extra[offset:, indices_extra.time_denominator] = element.denominator

                    if type(element) == KeySignature or type(element) == Key:
                        data.extra[offset:, indices_extra.num_sharps] = element.sharps

        target_file_path = os.path.join(target_dir, f'{chorale.metadata.number}.pt')
        torch.save({
            'data': data,
            'title': chorale.metadata.title
        }, target_file_path)

    # Move files to train / test directories
    file_paths = glob(os.path.join(target_dir, '*.pt'))
    random.shuffle(file_paths)  # Shuffle in place
    split_idx = int(len(file_paths) * split)
    for file_path in file_paths[split_idx:]:
        shutil.move(file_path, train_dir)
    for file_path in file_paths[:split_idx]:
        shutil.move(file_path, test_dir)

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
        'test': test_data_loader,
        'input_size': (60 + len(indices_parts)) * 4 + len(indices_extra),
        'output_sizes': (60 + len(indices_parts))
    }


def get_data_loaders(time_grid=0.25, root_dir=None, overwrite=False, split=0.15, batch_size=1, num_workers=1, context_radius=16):
    if root_dir is None:
        root_dir = os.path.join('.', 'data')

    data_dir = _generate_data(
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

    return EasyDict(data_loaders)
