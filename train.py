import datetime
import logging
import os
import pprint

import torch
import torch.nn
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import data
from model import BachNet


def main(config_passed):
    config = {
        'num_epochs': 100,
        'batch_size': 256,
        'hidden_size': 300,
        'use_cuda': True,
        'num_workers': 4,
        'lr': 0.001,
        'lr_step_size': 20,
        'lr_gamma': 0.95,
        'time_grid': 0.25,
        'context_radius': 16,
        'checkpoint_root_dir': os.path.join('.', 'checkpoints'),
        'checkpoint_interval': None,
        'log_interval': 10
    }

    # Save deviations from default config as string for logging
    blacklist = ['checkpoint_root_dir', 'checkpoint_interval', 'use_cuda', 'num_epochs', 'num_workers', 'log_interval']
    config_string = ' '.join([f'{k}={v}' for k, v in config_passed.items() if k not in blacklist]).strip()

    # Update default config with passed parameters
    config.update(config_passed)
    config = EasyDict(config)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(config.checkpoint_root_dir, f'{date} {config_string}')
    log_dir = os.path.join('.', 'runs', f'{date} {config_string}')
    writer = SummaryWriter(log_dir=log_dir)
    logging.basicConfig(level=logging.DEBUG)

    logging.debug(f'Configuration:\n{pprint.pformat(config)}')

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    logging.debug(f'Using device: {device}')

    logging.debug('Loading datasets...')
    data_loaders = data.get_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        time_grid=config.time_grid,
        context_radius=config.context_radius
    )

    logging.debug('Creating model...')
    model = BachNet(data_loaders['input_size'], data_loaders['output_sizes'], config.hidden_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    logging.debug('Training and testing...')
    for epoch in trange(config.num_epochs, unit='epoch'):
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()

            for batch_idx, batch in enumerate(data_loaders[phase]):
                inputs, targets = batch
                # inputs = torch.cat([v for v in inputs.values()], dim=2).to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(inputs)
                    losses = {k: criterion(predictions[k], targets[k].to(device)) for k in targets.keys()}
                    loss = sum(losses.values())

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if batch_idx % config.log_interval == 0 or len(data_loaders[phase]) / config.batch_size < config.log_interval:
                    step = int((float(epoch) + (batch_idx / len(data_loaders[phase]))) * 1000)
                    writer.add_scalars('loss', {phase: loss.item()}, step)

        lr_scheduler.step()

        if config.checkpoint_interval is not None and (epoch + 1) % config.checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'{str(epoch + 1).zfill(4)} {config_string}.pt')
            torch.save({
                'config': config,
                'state': model.state_dict(),
                'epoch': epoch
            }, checkpoint_path)


if __name__ == '__main__':
    config = {
        'num_epochs': 10000,
        'batch_size': 512,
        'hidden_size': 500,
        'context_radius': 32,
        'lr': 0.001,
        'lr_step_size': 50,
        'log_interval': 1,
        # 'checkpoint_interval': 5
    }
    main(config)
