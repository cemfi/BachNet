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
from model import BachNetTraining


def main(config):
    logging.debug('Initializing...')

    # Prepare logging
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(config.checkpoint_root_dir, f'{date} {str(config)}')
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
        context_radius=config.context_radius
    )

    logging.debug('Creating model...')
    model = BachNetTraining(
        hidden_size=config.hidden_size,
        context_radius=config.context_radius
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    logging.debug('Training and testing...')
    loss_per_epoch = {'train': [], 'test': []}
    # for epoch in range(config.num_epochs):
    for epoch in trange(config.num_epochs, unit='epoch'):

        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            loss_list = []

            with torch.set_grad_enabled(phase == 'train'):

                for batch_idx, batch in enumerate(data_loaders[phase]):
                    inputs, targets = batch
                    # Transfer to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    targets = {k: v.to(device) for k, v in targets.items()}

                    predictions = model(inputs)
                    losses = {k: criterion(predictions[k], targets[k]) for k in targets.keys()}
                    loss = sum(losses.values())
                    loss_list.append(loss.item())

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Log current loss
                    if batch_idx % config.log_interval == 0:
                        step = int((float(epoch) + (batch_idx / len(data_loaders[phase]))) * 1000)
                        writer.add_scalars('loss', {phase: loss.item()}, step)
                        writer.add_scalars('loss_per_parts', {f'{phase}_{k}': v for k, v in losses.items()}, step)

                # Log mean loss per epoch
                mean_loss_per_epoch = mean(loss_list)
                loss_per_epoch[phase].append(mean_loss_per_epoch)
                writer.add_scalars('loss', {phase + '_mean': mean_loss_per_epoch}, (epoch + 1) * 1000)
                writer.file_writer.flush()

        lr_scheduler.step()

        if config.checkpoint_interval is not None and (epoch + 1) % config.checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'{str(epoch + 1).zfill(4)} {str(config)}.pt')
            torch.save({
                'config': config,
                'state': model.state_dict(),
                'epoch': epoch,
                'losses_train': loss_per_epoch['train'],
                'losses_test': loss_per_epoch['test'],
            }, checkpoint_path)

    min_test_loss = min(loss_per_epoch['test'])
    min_test_loss_idx = loss_per_epoch['test'].index(min_test_loss)
    logging.info(f'Lowest testing loss after epoch {min_test_loss_idx}: {min_test_loss}')

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    configs = []
    for hidden_size in [350, 400, 450]:
        config = utils.Config({
            'num_epochs': 300,
            'batch_size': 8192,
            'num_workers': 4,
            'hidden_size': hidden_size,
            'context_radius': 32,
            'time_grid': 0.25,
            'lr': 0.001,
            'lr_gamma': 0.98,
            'lr_step_size': 10,
            'checkpoint_interval': 10
        })
        configs.append(config)

    from tqdm import tqdm

    for config in tqdm(configs):
        main(config)

    # from concurrent.futures import ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     executor.map(main, configs)
