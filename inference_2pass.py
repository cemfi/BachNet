import torch

import data
import utils
from model import BachNetInference


def main(soprano_path, checkpoint_path):
    # Disable autograd to save a huge amount of memory
    torch.set_grad_enabled(False)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state = checkpoint['state']

    model = BachNetInference(
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

    length = inputs['soprano'].shape[0]

    # Empty "history" for generated parts
    inputs['bass'] = torch.zeros((config.context_radius, data.part_size))

    # Zero padding for input data
    for part in ['soprano', 'extra']:
        inputs[part] = torch.cat([
            torch.zeros((config.context_radius, inputs[part].shape[1])),
            inputs[part],
            torch.zeros((config.context_radius, inputs[part].shape[1]))
        ], dim=0)

    outputs = {k: [] for k in inputs.keys()}

    for step in range(length):
        prediction = model({
            'soprano': inputs['soprano'][step:step + 2 * config.context_radius + 1],
            'bass': inputs['bass'],
            'extra': inputs['extra'][step:step + 2 * config.context_radius + 1]
        })
        bass = prediction.items()
        outputs['bass'].append(bass)
        inputs['bass'] = torch.cat([inputs['bass'][1:], bass], dim=0)
        outputs['soprano'].append(inputs['soprano'][step + config.context_radius].unsqueeze(dim=0))
        outputs['extra'].append(inputs['extra'][step + config.context_radius].unsqueeze(dim=0))

    outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    score = utils.tensors_to_stream(outputs, config, metadata)
    score.show('musicxml')
    score.write('musicxml', 'output.musicxml')


if __name__ == '__main__':
    import os
    from glob import glob
    #latest_checkpoint = sorted(glob('./checkpoints/**/*.pt'))[-1]
    #print(os.path.split(latest_checkpoint)[-1])

    main(
        soprano_path='./data/musicxml/127_soprano.musicxml',
        checkpoint_path='./checkpoints/0300 batch_size=8192 hidden_size=600 context_radius=32 time_grid=0.25 lr=0.001 lr_gamma=0.98 lr_step_size=10 split=0.05.pt'
        # checkpoint_path=latest_checkpoint
    )