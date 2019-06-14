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
        split=config.split,
    )

    logging.debug('Creating model...')
    model_continuo = BachNetTrainingContinuo(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius,
    ).to(device)
    model_middle_parts = BachNetTrainingMiddleParts(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius
    ).to(device)
    params_continuo = [p for p in model_continuo.parameters() if p.requires_grad]
    params_middleparts = [p for p in model_middle_parts.parameters() if p.requires_grad]
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
            model_middle_parts.train() if phase == 'train' else model_middle_parts.eval()
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
                    inputs_for_middle_parts = {k: inputs[k].to(device) for k in ['soprano', 'alto', 'tenor', 'bass_with_context', 'extra']}
                    targets_continuo = {k: targets[k].to(device) for k in ['bass']}
                    targets_middleparts = {k: targets[k].to(device) for k in ['alto', 'tenor']}

                    predictions_continuo = model_continuo(inputs_for_continuo)
                    losses_continuo = {k: criterion(predictions_continuo[k], targets_continuo[k]) for k in targets_continuo.keys()}

                    predictions_middleparts = model_middle_parts(inputs_for_middle_parts)
                    losses_middleparts = {k: criterion(predictions_middleparts[k], targets_middleparts[k]) for k in targets_middleparts.keys()}

                    loss = sum([sum(losses_middleparts.values()), sum(losses_continuo.values())])

                    loss_lists['all'].append(loss.item())
                    for k in losses_continuo.keys():
                        loss_lists[k].append(losses_continuo[k].item())
                    for k in losses_middleparts.keys():
                        loss_lists[k].append(losses_middleparts[k].item())

                    if phase == 'train':
                        optimizer_continuo.zero_grad()
                        optimizer_middleparts.zero_grad()
                        sum(losses_continuo.values()).backward()
                        sum(losses_middleparts.values()).backward()
                        # loss.backward()
                        optimizer_continuo.step()
                        optimizer_middleparts.step()

                    # Log current loss
                    if batch_idx % config.log_interval == 0:
                        step = int((float(epoch) + (batch_idx / len(data_loaders[phase]))) * 1000)
                        writer.add_scalars('loss', {phase: loss.item()}, step)
                        writer.add_scalars('loss_per_parts', {f'{phase}_{k}': v for k, v in losses_continuo.items()}, step)
                        writer.add_scalars('loss_per_parts', {f'{phase}_{k}': v for k, v in losses_middleparts.items()}, step)

                # Log mean loss per epoch
                mean_loss_per_epoch = mean(loss_lists['all'])
                writer.add_scalars('loss', {phase + '_mean': mean_loss_per_epoch}, (epoch + 1) * 1000)
                writer.file_writer.flush()

        lr_scheduler_continuo.step()
        lr_scheduler_middleparts.step()

        if config.checkpoint_interval is not None and (epoch + 1) % config.checkpoint_interval == 0:
            subfolder = f'{date} {str(config)}'
            os.makedirs(os.path.join(config.checkpoint_root_dir, subfolder), exist_ok=True)
            checkpoint_path = os.path.join(config.checkpoint_root_dir, subfolder, f'{date} {str(config)} {str(epoch + 1).zfill(4)}.pt')
            torch.save({
                'config': config,
                'state_continuo': model_continuo.state_dict(),
                'state_middle_parts': model_middle_parts.state_dict(),
                'epoch': epoch,
                'loss_bass': mean(loss_lists['bass']),
                'loss_alto': mean(loss_lists['alto']),
                'loss_tenor': mean(loss_lists['tenor'])

            }, checkpoint_path)

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    configs = []
    params = [
        (32, 700)
    ]
    for radius, hidden_size in params:
        config = utils.Config({
            'num_epochs': 1000,
            'batch_size': 8192,
            'num_workers': 4,
            'hidden_size': hidden_size,
            'context_radius': radius,
            'time_grid': 0.25,
            'lr': 0.001,
            'lr_gamma': 0.98,
            'lr_step_size': 10,
            'checkpoint_interval': 10,
            'split': 0.05,

        })
        configs.append(config)

    from tqdm import tqdm

    for config in tqdm(configs):
        main(config)
