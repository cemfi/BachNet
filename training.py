import datetime
import logging
import os
import pprint
from statistics import mean

import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import data
import utils
from model import BachNetTrainingContinuo, BachNetTrainingMiddleParts


def main(config):
    logging.debug('Initializing...')

    # Prepare logging
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('.', 'runs', f'{date} {str(config)}')
    writer = SummaryWriter(log_dir=log_dir)

    logging.debug(f'Configuration:\n{pprint.pformat(config)}')

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    logging.debug(f'Using device: {device}')

    logging.debug('Loading datasets...')
    data_loaders = data.get_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        time_grid=config.time_grid,
        context_radius=config.context_radius,
        transpositions=config.transpositions,
        split=config.split,
    )

    logging.debug('Creating model...')
    model_continuo = BachNetTrainingContinuo(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius
    ).to(device)
    model_middleparts = BachNetTrainingMiddleParts(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius
    ).to(device)
    params_continuo = [p for p in model_continuo.parameters() if p.requires_grad]
    params_middleparts = [p for p in model_middleparts.parameters() if p.requires_grad]
    optimizer_continuo = torch.optim.Adam(params_continuo, lr=config.lr)
    optimizer_middleparts = torch.optim.Adam(params_middleparts, lr=config.lr)
    lr_scheduler_continuo = torch.optim.lr_scheduler.StepLR(
        optimizer_continuo,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )
    lr_scheduler_middleparts = torch.optim.lr_scheduler.StepLR(
        optimizer_middleparts,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    logging.debug('Training and testing...')
    # for epoch in range(config.num_epochs):
    for epoch in trange(config.num_epochs, unit='epoch'):

        for phase in ['train', 'test']:
            model_continuo.train() if phase == 'train' else model_continuo.eval()
            model_middleparts.train() if phase == 'train' else model_middleparts.eval()
            loss_lists = {
                'all': [],
                'bass': [],
                'alto': [],
                'tenor': []
            }

            with torch.set_grad_enabled(phase == 'train'):

                for batch_idx, batch in enumerate(data_loaders[phase]):
                    inputs, targets = batch
                    # Transfer to device
                    inputs_for_continuo = {k: inputs[k].to(device) for k in ['soprano', 'bass', 'extra']}
                    inputs_for_middleparts = {k: inputs[k].to(device) for k in ['soprano', 'alto', 'tenor', 'bass_withcontext', 'extra']}
                    targets = {k: targets[k].to(device) for k in ['soprano', 'bass']}

                    predictions = model_continuo(inputs_for_continuo)
                    losses = {k: criterion(predictions[k], targets[k]) for k in targets.keys()}

                    loss = sum(losses.values())

                    #loss_lists['all'].append(loss.item())
                    #for k in losses.keys():
                    #    loss_lists[k].append(losses[k].item())

                    predictions = model_middleparts(inputs_for_middleparts)
                    losses = {k: criterion(predictions[k], targets[k]) for k in targets.keys()}

                    if phase == 'train':
                        optimizer_continuo.zero_grad()
                        optimizer_middleparts.zero_grad()
                        loss.backward()
                        optimizer_continuo.step()
                        optimizer_middleparts.step()

                    # Log current loss
                    if batch_idx % config.log_interval == 0:
                        step = int((float(epoch) + (batch_idx / len(data_loaders[phase]))) * 1000)
                        writer.add_scalars('loss', {phase: loss.item()}, step)
                        writer.add_scalars('loss_per_parts', {f'{phase}_{k}': v for k, v in losses.items()}, step)

                # Log mean loss per epoch
                mean_loss_per_epoch = mean(loss_lists['all'])
                writer.add_scalars('loss', {phase + '_mean': mean_loss_per_epoch}, (epoch + 1) * 1000)
                writer.file_writer.flush()

        lr_scheduler_continuo.step()
        lr_scheduler_middleparts.step()

        if config.checkpoint_interval is not None and (epoch + 1) % config.checkpoint_interval == 0:
            os.makedirs(config.checkpoint_root_dir, exist_ok=True)
            checkpoint_path = os.path.join(config.checkpoint_root_dir, f'{date} {str(config)} {str(epoch + 1).zfill(4)}.pt')
            torch.save({
                'config': config,
                'state': model.state_dict(),
                'epoch': epoch,
                'loss_bass': mean(loss_lists['bass']),
                'loss_alto': mean(loss_lists['alto']),
                'loss_tenor': mean(loss_lists['tenor'])

            }, checkpoint_path)

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    configs = []
    for hidden_size in [600]:
        config = utils.Config({
            'num_epochs': 200,
            'batch_size': 8192,
            'num_workers': 4,
            'hidden_size': hidden_size,
            'context_radius': 32,
            'time_grid': 0.25,
            'lr': 0.001,
            'lr_gamma': 0.95,
            'lr_step_size': 10,
            'checkpoint_interval': 10,
            'split': 0.05,
        })
        configs.append(config)

    from tqdm import tqdm

    for config in tqdm(configs):
        main(config)
