#!/usr/bin/env python3

import os
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tensorboardX import SummaryWriter

from bach_data import BachDataset
from bach_net import BachNet3
from bach_datadownloader_new import DataDownloader


parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning_rate", default=0.001, help="learning rate", type=float)
parser.add_argument("-g", "--gamma", default=0.9, help="gamma for decaying lr", type=float)
parser.add_argument("-hs", "--hidden_size", default=1072, help="hidden layer size", type=int)
parser.add_argument("-nh", "--number_hidden", default=2, help="number hidden layers", type=int)
parser.add_argument("-fs", "--frame_size", default=16, help="frame size", type=int)
parser.add_argument("-do", "--dropout", default=0.5, help="dropout", type=float)
parser.add_argument("-ne", "--epochs", default=25, help="number of epochs", type=int)
parser.add_argument("-co", "--comment", default='', help="add a comment", type=str)

args = parser.parse_args()

date = "{0:%m-%d %H-%M}".format(datetime.datetime.now())
idString = date + "-lr" + str(args.learning_rate) + "-g" + str(args.gamma) + "-hs" + str(args.hidden_size) + "-nh" + str(args.number_hidden) + "-fs" + str(args.frame_size) + "-do" + str(args.dropout) + "-" + args.comment

data_path = os.path.join('.', 'data')
if not os.path.exists(data_path):
    downloader = DataDownloader(data_path, transpositions=[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], overwrite=True)
    downloader.download(valPercent=15)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

batchSize = 500

# Parameters
params = {'batch_size': batchSize, 'shuffle': True}

# Datasets
dataloaders = {
    'train': DataLoader(BachDataset(os.path.join(data_path, 'train'), args.frame_size), **params),
    'valid': DataLoader(BachDataset(os.path.join(data_path, 'valid'), args.frame_size), **params)
}

model = BachNet3(input_dims=279, hidden_dims=args.hidden_size, output_dims=279, num_hidden_layers=args.number_hidden, dropout=args.dropout).to(device)  # former hidden 600 - wasnt better?
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)


writer = SummaryWriter(log_dir=f'runs/{idString}')

tqdm.write(idString)
for epoch in tqdm(range(args.epochs), unit='epoch'):
    for phase in ['train', 'valid']:
        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(tqdm(dataloaders[phase], desc=phase, unit='batch')):
            batch, labels = batch_sample
            batch, labels = batch.to(device), labels.to(device)
            batch = batch.permute(2, 0, 1)
            labels = labels.permute(2, 0, 1)

            with torch.set_grad_enabled(phase == 'train'):
                h = model.init_hidden(len(batch[0]))
                y_pred, hidden = model(batch, h)

                loss = criterion(y_pred, labels)

                step = int((float(epoch) + (batch_idx / len(dataloaders[phase]))) * 1000)
                writer.add_scalars('loss', {phase: loss.item()}, step)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    directory = os.path.join('.', 'checkpoints', idString)
    os.makedirs(directory, exist_ok=True)

    tempName = os.path.join(directory, idString + str(epoch) + ".pt")
    torch.save(model.state_dict(), tempName)
