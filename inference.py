import logging
from copy import deepcopy
from datetime import datetime

import torch
from easydict import EasyDict
from music21 import stream, clef
from music21.expressions import Fermata
from music21.key import KeySignature
from music21.layout import StaffGroup
from music21.metadata import Metadata
from music21.meter import TimeSignature
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.tie import Tie

import data
from model import BachNetInference


def main(soprano_path, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = EasyDict(checkpoint['config'])
    state = checkpoint['state']

    model = BachNetInference(
        batch_size=1,
        hidden_size=config.hidden_size,
        context_radius=config.context_radius
    )
    model.load_state_dict(state)
    model.eval()

    sample = data.generate_data_inference(
        time_grid=config.time_grid,
        soprano_path=soprano_path
    )

    inputs = sample['data']
    metadata = sample['metadata']

    length = inputs.soprano.shape[0]

    # Empty "history" for generated parts
    for part in ['alto', 'tenor', 'bass']:
        inputs[part] = torch.zeros((config.context_radius, inputs.soprano.shape[1]))

    # Zero padding for input data
    for part in ['soprano', 'extra']:
        inputs[part] = torch.cat([
            torch.zeros((config.context_radius, inputs[part].shape[1])),
            inputs[part],
            torch.zeros((config.context_radius, inputs[part].shape[1]))
        ], dim=0)

    outputs = EasyDict({
        'soprano': [],
        'alto': [],
        'tenor': [],
        'bass': [],
        'extra': [],
    })

    for step in range(length):
        predictions = model({
            'soprano': inputs.soprano[step:step + 2 * config.context_radius + 1],
            'alto': inputs.alto,
            'tenor': inputs.tenor,
            'bass': inputs.bass,
            'extra': inputs.extra[step:step + 2 * config.context_radius + 1]
        })
        for part_name, one_hot in predictions.items():
            outputs[part_name].append(one_hot)
            inputs[part_name] = torch.cat([inputs[part_name][1:], one_hot], dim=0)
        outputs.soprano.append(inputs.soprano[step + config.context_radius].unsqueeze(dim=0))
        outputs.extra.append(inputs.extra[step + config.context_radius].unsqueeze(dim=0))

    outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    cur_measure_number = 0
    parts = {}
    for part_name in outputs.keys():
        if part_name == 'extra':
            continue
        part = stream.Part(id=part_name)
        parts[part_name] = part

    last_time_numerator = None
    last_time_denominator = None
    last_num_sharps = None
    for step in range(length):
        extra = outputs['extra'][step]
        cur_time_numerator = int(extra[data.indices_extra.time_numerator].item())
        cur_time_denominator = int(extra[data.indices_extra.time_denominator].item())
        cur_num_sharps = int(extra[data.indices_extra.num_sharps].item())
        cur_time_pos = extra[data.indices_extra.time_pos].item()
        has_fermata = extra[data.indices_extra.has_fermata].item() == 1

        if cur_time_pos == 1.0 or cur_measure_number == 0:
            for part_name, part in parts.items():
                part.append(stream.Measure(number=cur_measure_number))
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
            if idx == data.indices_parts.is_continued:
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
            elif idx == data.indices_parts.is_rest:
                part[-1].append(Rest(quarterLength=config.time_grid))
            else:
                pitch = Pitch()
                part[-1].append(Note(pitch, quarterLength=config.time_grid))
                # Set pitch value AFTER appending to measure in order to avoid unnecessary accidentals
                pitch.midi = idx + (data.pitch_size // 2) - len(data.indices_parts)

        if has_fermata:
            for part in parts.values():
                fermata = Fermata()
                fermata.type = 'upright'
                part[-1][-1].expressions.append(fermata)

    score = stream.Score()
    score.append(Metadata())
    score.metadata.title = f"{metadata.title} ({metadata.number})"
    score.metadata.composer = f"Melody: {metadata.composer}\nArrangement: BachNet ({datetime.now().year})"
    for part in parts.values():
        part[-1].rightBarline = 'light-heavy'
        score.append(part)

    score.stripTies(inPlace=True, retainContainers=True)
    staff_group = StaffGroup(list(parts.values()), symbol='bracket')
    staff_group.barTogether = 'yes'
    score.append(staff_group)
    score.show('musicxml')


if __name__ == '__main__':
    main(
        soprano_path='./data/musicxml/008_soprano.musicxml',
        checkpoint_path='./checkpoints/2019-06-03_02-59-46 hidden_size=79 context_radius=32 time_grid=0.25/0035 hidden_size=79 context_radius=32 time_grid=0.25.pt'
    )
