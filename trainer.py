import os
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import BachDataset
from model import BachNet


def train(config):
    date = "{0:%m-%d %H-%M}".format(datetime.datetime.now())
    idString = date + "-lr" + str(config['learning_rate']) + "-g" + str(config['learning_gamma']) +"-ls" + str(config['learning_step']) + "-hs" + str(config['hidden_size']) + "-nh" + str(config['number_hidden']) + "-fs" + str(config['frame_size']) + "-do" + str(config['dropout']) + '-'
    checkpoint_dir = os.path.join('.', 'checkpoints', idString)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f'runs/{idString}')

    data_path = 'data/'


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    params = {'batch_size': 50, 'shuffle': True}

    # Datasets
    dataloaders = {
        'train':
            DataLoader(
                BachDataset(
                    os.path.join(data_path, 'train'), config['frame_size']
                ), **params
            ),
        'valid':
            DataLoader(
                BachDataset(
                    os.path.join(data_path, 'valid'), config['frame_size']
                ), **params
            )
    }

    model = BachNet(
        input_dims=279,
        hidden_dims=config['hidden_size'],
        output_dims=62,
        num_hidden_layers=config['number_hidden'],
        dropout=config['dropout']
    ).to(device)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(
        optimizer,
        step_size=config['learning_step'],
        gamma=config['learning_gamma']
    )

    for epoch in tqdm(range(config['number_epochs']), unit='epoch'):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(dataloaders[phase]):
                batch, labels = batch_sample
                batch, labels = batch.to(device), labels.to(device)
                batch = batch.permute(2, 0, 1)
                labels = labels.permute(2, 0, 1)

                with torch.set_grad_enabled(phase == 'train'):
                    h = model.init_hidden(len(batch[0]))
                    y_pred_B, y_pred_A, y_pred_T, hidden = model(batch, h)

                    current_batchsize = y_pred_A[0, :, 0].shape[0]
                    beforepad = torch.zeros(config['frame_size'], current_batchsize, 8)          # padding for format
                    afterpad = torch.zeros(config['frame_size'], current_batchsize, 85)        # padding for format

                    y_pred = torch.cat((beforepad, y_pred_B, y_pred_T, y_pred_A, afterpad), 2)

                    loss = criterion(y_pred, labels)

                    step = int(
                        (float(epoch) + (batch_idx / len(dataloaders[phase]))) *
                        1000
                    )
                    writer.add_scalars('loss', {phase: loss.item()}, step)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        if config['save_checkpoints']:
            tempName = os.path.join(checkpoint_dir, idString + str(epoch) + ".pt")
            torch.save({'state': model.state_dict(), 'config': config}, tempName)


if __name__ == '__main__':
    from random import choice, randint

    config = {
        'learning_rate': 0.001,
        'learning_gamma': 0.9,
        'learning_step': 3,
        'hidden_size': 1115,
        'number_hidden': 2,
        'frame_size': 16,
        'dropout': 0.5,
        'number_epochs': 1,
        'save_checkpoints': True,
    }

    train(config)
