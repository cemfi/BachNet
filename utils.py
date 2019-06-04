import logging
import os
from copy import deepcopy
from datetime import datetime

import torch
from music21 import clef
from music21.expressions import Fermata
from music21.key import KeySignature
from music21.metadata import Metadata
from music21.meter import TimeSignature
from music21.note import Rest, Note
from music21.pitch import Pitch
from music21.stream import Score, Part, Measure
from music21.tie import Tie

from data import indices_extra, indices_parts, pitch_size


class Config(object):
    num_epochs = 100
    batch_size = 128
    hidden_size = 75
    use_cuda = True
    num_workers = 1
    lr = 0.0001
    lr_step_size = 10
    lr_gamma = 0.95
    time_grid = 0.25
    context_radius = 16
    checkpoint_root_dir = os.path.join('.', 'checkpoints')
    checkpoint_interval = None
    log_interval = 1

    def __init__(self, config=None):
        if config is not None:
            self.explicit = config
        else:
            self.explicit = {}
        self.__dict__.update(config)

    def __repr__(self):
        blacklist = ['checkpoint_root_dir', 'checkpoint_interval', 'use_cuda', 'num_epochs', 'num_workers', 'log_interval']
        config_string = ' '.join([f'{k}={v}' for k, v in self.explicit.items() if k not in blacklist]).strip()
        return config_string


def generate_txt_output(data, path):
    with open(path, 'w') as fp:
        for pitches in data.t().flip(dims=[0]):
            line = ''
            for step in pitches:
                char = '*' if step == 1 else ' '
                line += char
            fp.write(line)
            fp.write('\n')


def tensors_to_stream(outputs, config, metadata):
    cur_measure_number = 0
    parts = {}
    for part_name in outputs.keys():
        if part_name == 'extra':
            continue
        part = Part(id=part_name)
        parts[part_name] = part

    last_time_numerator = None
    last_time_denominator = None
    last_num_sharps = None
    for step in range(outputs['soprano'].shape[0]):
        extra = outputs['extra'][step]
        cur_time_numerator = int(extra[indices_extra['time_numerator']].item())
        cur_time_denominator = int(extra[indices_extra['time_denominator']].item())
        cur_num_sharps = int(extra[indices_extra['num_sharps']].item())
        cur_time_pos = extra[indices_extra['time_pos']].item()
        has_fermata = extra[indices_extra['has_fermata']].item() == 1

        if cur_time_pos == 1.0 or cur_measure_number == 0:
            for part_name, part in parts.items():
                part.append(Measure(number=cur_measure_number))
                if cur_measure_number == 0:
                    if part_name in ['soprano', 'alto']:
                        part[-1].append(clef.TrebleClef())
                    else:
                        part[-1].append(clef.BassClef())
            cur_measure_number += 1

        if last_time_numerator is None or last_time_denominator is None or cur_time_numerator != last_time_numerator or cur_time_denominator != last_time_denominator:
            for part in parts.values():
                part[-1].append(TimeSignature(f'{cur_time_numerator}/{cur_time_denominator}'))
            last_time_numerator = cur_time_numerator
            last_time_denominator = cur_time_denominator

        if last_num_sharps is None or cur_num_sharps != last_num_sharps:
            for part in parts.values():
                part[-1].append(KeySignature(cur_num_sharps))
            last_num_sharps = cur_num_sharps

        for part_name, part in parts.items():
            idx = torch.argmax(outputs[part_name][step]).item()
            if idx == indices_parts['is_continued']:
                try:
                    last_element = part[-1].flat.notesAndRests[-1]
                    cur_element = deepcopy(last_element)
                    if last_element.tie is not None and last_element.tie.type == 'stop':
                        last_element.tie = Tie('continue')
                    else:
                        last_element.tie = Tie('start')
                    cur_element.tie = Tie('stop')
                except IndexError:
                    logging.debug('Warning: "is_continued" on first beat. Replaced by rest.')
                    cur_element = Rest(quarterLength=config.time_grid)
                part[-1].append(cur_element)
            elif idx == indices_parts['is_rest']:
                part[-1].append(Rest(quarterLength=config.time_grid))
            else:
                pitch = Pitch()
                part[-1].append(Note(pitch, quarterLength=config.time_grid))
                # Set pitch value AFTER appending to measure in order to avoid unnecessary accidentals
                pitch.midi = idx + (pitch_size // 2) - len(indices_parts)

        if has_fermata:
            for part in parts.values():
                fermata = Fermata()
                fermata.type = 'upright'
                part[-1][-1].expressions.append(fermata)

    score = Score()
    score.append(Metadata())
    score.metadata.title = f"{metadata.title} ({metadata.number})"
    score.metadata.composer = f"Melody: {metadata.composer}\nArrangement: BachNet ({datetime.now().year})"
    for part in parts.values():
        part[-1].rightBarline = 'light-heavy'

    score.append(parts['soprano'])
    score.append(parts['alto'])
    score.append(parts['tenor'])
    score.append(parts['bass'])

    score.stripTies(inPlace=True, retainContainers=True)

    return score
