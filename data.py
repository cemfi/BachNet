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
from music21.key import Key, KeySignature
from music21.meter import TimeSignature
from music21.note import Note, Rest
from torch.utils.data import Dataset, RandomSampler, BatchSampler, SequentialSampler, DataLoader

log = {
    'total': 0,
    'total_incl_aug': 0,
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
    '10': 0,
    '11': 0,
    '3/4': 0,
    '4/4': 0,
    '3/2': 0
}

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
    'pitch_offset': 5,
    'has_sharps_0': 6,
    'has_sharps_1': 7,
    'has_sharps_2': 8,
    'has_sharps_3': 9,
    'has_sharps_4': 10,
    'has_sharps_5': 11,
    'has_sharps_6': 12,
    'has_sharps_7': 13,
    'has_sharps_8': 14,
    'has_sharps_9': 15,
    'has_sharps_10': 16,
    'has_sharps_11': 17
}

trans = 0

min_pitches = {
    'bass': 36 - trans,
    'tenor': 48 - trans,
    'alto': 53 - trans,
    'soprano': 57 - trans
}

max_pitches = {
    'bass': 64 + trans,
    'tenor': 69 + trans,
    'alto': 74 + trans,
    'soprano': 81 + trans
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
                self.data[part_name].append(
                    torch.cat([part_data, torch.zeros((context_radius, part_data.shape[1]))], dim=0))

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
    keys = list(stream.flat.getElementsByClass(Key))
    if len(keys) > 0:
        num_sharps = keys[0].sharps
        num_sharps = (num_sharps + 12) % 12
    else:
        key_sigs = list(stream.flat.getElementsByClass(KeySignature))
        if len(key_sigs) > 0:
            num_sharps = key_sigs[0].sharps
            num_sharps = (num_sharps + 12) % 12
        else:
            num_sharps = 0

    data['extra'][:, num_sharps + indices_extra['has_sharps_0']] = 1

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
    }


def _generate_data_training(time_grid, root_dir, overwrite, split, debug):
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

    for chorale in chorales.Iterator(returnType='stream'):
        print(f'Converting {chorale.corpusFilepath}')

        ts_not_yet_logged = True
        last_logged_ts = ''

        # Use only 10 files when debugging
        if debug and len(chorale_numbers) == 10:
            break

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

        keys = list(chorale.flat.getElementsByClass(Key))
        if len(keys) > 0:
            num_sharps = keys[0].sharps
            num_sharps = (num_sharps + 12) % 12
        else:
            key_sigs = list(chorale.flat.getElementsByClass(KeySignature))
            if len(key_sigs) > 0:
                num_sharps = key_sigs[0].sharps
                num_sharps = (num_sharps + 12) % 12
            else:
                num_sharps = 0

        log[str(num_sharps)] = log[str(num_sharps)] + 1
        log['total'] = log['total'] + 1
        # Save soprano in own file for inference
        chorale['Soprano'].write('musicxml', os.path.join(musicxml_dir,
                                                          f'{str(chorale.metadata.number).zfill(3)}_soprano.musicxml'))
        chorale.write('musicxml', os.path.join(musicxml_dir, f'{str(chorale.metadata.number).zfill(3)}_full.musicxml'))

        # # Get minimum and maximum transpositions
        transpositions_down = -float('inf')
        transpositions_up = float('inf')
        for part_name, part in streams.items():
            min_pitch, max_pitch = ambitus.getPitchSpan(part)
            transpositions_down = max(transpositions_down, min_pitches[part_name] - min_pitch.midi)
            transpositions_up = min(transpositions_up, max_pitches[part_name] - max_pitch.midi)
        # transpositions_down = -trans
        # transpositions_up = trans

        length = math.ceil(streams['soprano'].highestTime / time_grid)
        for t in range(transpositions_down, transpositions_up + 1):
            log['total_incl_aug'] = log['total_incl_aug'] + 1
            # print(transpositions_down)
            # print(transpositions_up)
            data = {'extra': torch.zeros((length, len(indices_extra)))}
            # Note transposition offset
            data['extra'][:, indices_extra['pitch_offset']] = t
            cur_sharps = (num_sharps + (t * 7)) % 12
            data['extra'][:, cur_sharps + indices_extra['has_sharps_0']] = 1

            for part_name, part in streams.items():
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
                                if t == 0:
                                    log['3/4'] = log['3/4'] + 1
                                    if not ts_not_yet_logged:
                                        print("2nd ts found")
                                        print(last_logged_ts)
                                        print(' and 3/4')
                                        streams['soprano'].show()
                                    ts_not_yet_logged = False
                                    last_logged_ts = '3/4'
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
                            elif element.ratioString == '4/4':
                                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 1
                                if t == 0:
                                    log['4/4'] = log['4/4'] + 1
                                    if not ts_not_yet_logged:
                                        print("2nd ts found")
                                        print(last_logged_ts)
                                        print(' and 4/4')
                                    ts_not_yet_logged = False
                                    last_logged_ts = '4/4'
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 0
                            elif element.ratioString == '3/2':
                                data['extra'][offset:, indices_extra['has_time_signature_3/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_4/4']] = 0
                                data['extra'][offset:, indices_extra['has_time_signature_3/2']] = 1
                                if t == 0:
                                    log['3/2'] = log['3/2'] + 1
                                    if not ts_not_yet_logged:
                                        print("2nd ts found")
                                        print(last_logged_ts)
                                        print(' and 3/2')
                                    ts_not_yet_logged = False
                                    last_logged_ts = '3/2'

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

            target_file_path = os.path.join(target_dir,
                                            f'{str(chorale.metadata.number).zfill(3)}{signum}{int(math.fabs(t))}.pt')
            torch.save({
                'data': data,
                'title': chorale.metadata.title
            }, target_file_path)

        ts_not_yet_logged = False

        # print(log)
        chorale_numbers.append(str(chorale.metadata.number).zfill(3))

    # Move files to train / test directories
    random.shuffle(chorale_numbers)  # Shuffle in place
    split_idx = int(len(chorale_numbers) * split)
    split_idx = max(1, split_idx)

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


def get_data_loaders(time_grid=0.25, root_dir=None, overwrite=False, split=0.05, batch_size=1, num_workers=1,
                     context_radius=32, debug=False):
    '''
    Gets the data loaders.

    Args:
        time_grid:
        root_dir:
        overwrite:
        split:
        batch_size:
        num_workers:
        context_radius:
        debug: load only few files for testing

    Returns:

    '''
    if root_dir is None:
        root_dir = os.path.join('.', 'data')

    data_dir = _generate_data_training(
        time_grid=time_grid,
        root_dir=root_dir,
        overwrite=overwrite,
        split=split,
        debug=debug
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
