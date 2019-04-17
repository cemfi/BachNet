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
from bach_datadownloader_new import DataDownloader2

# hyperparameters

learningRate = 0.001
gamma = 0.8
hiddenSize = 278
numberHidden = 10
frameSize = 32
dropout = 0.5
comment = ""

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learningRate", help="learning rate", type=float)
parser.add_argument("-g", "--gamma", help="gamma for decaying lr", type=float)
parser.add_argument("-hs", "--hiddenSize", help="hidden layer size", type=int)
parser.add_argument("-nh", "--numberHidden", help="number hidden layers", type=int)
parser.add_argument("-fs", "--frameSize", help="frame size", type=int)
parser.add_argument("-do", "--dropout", help="dropout", type=float)

parser.add_argument("-co", "--comment", help="add a comment", type=str)

args = parser.parse_args()
if args.learningRate is not None:
    learningRate = args.learningRate
if args.gamma:
    gamma = args.gamma
if args.hiddenSize:
    hiddenSize = args.hiddenSize
if args.numberHidden:
    numberHidden = args.numberHidden
if args.frameSize:
    frameSize = args.frameSize
if args.dropout:
    dropout = args.dropout
if args.comment:
    comment = args.comment


date = "{0:%m-%d %H-%M}".format(datetime.datetime.now())
idString = date + "-lr" + str(learningRate) + "-g" + str(gamma) + "-hs" + str(hiddenSize) + "-nh" + str(numberHidden) + "-fs" + str(frameSize) + "-do" + str(dropout) + "-" + comment

print(idString)

data_path = os.path.join('.', 'chordDataSQ')
if not os.path.exists(data_path):
    downloader = DataDownloader2(data_path, [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], True)
    downloader.download(valPercent=15)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

batchSize = 40

# Parameters
params = {'batch_size': batchSize, 'shuffle': True}  # 'num_workers': 2}    #doesnt work: file open in getitem? google

# Datasets
dataloaders = {
    'train': DataLoader(BachDataset(os.path.join(data_path, 'train'), frameSize, batchSize), **params),
    'valid': DataLoader(BachDataset(os.path.join(data_path, 'valid'), frameSize, batchSize), **params)
}

model = BachNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden, dropout=dropout).to(device)  # former hidden 600 - wasnt better?
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
scheduler = StepLR(optimizer, step_size=3, gamma=gamma)

num_epochs = 20
lossLog = []
lossLogVal = []

writer = SummaryWriter(log_dir=f'runs/{idString}')

for epoch in range(num_epochs):
    for phase in ['train', 'valid']:
        tqdm.write('='*80)
        tqdm.write(f'Epoch {epoch}')
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

                lossLog.append(loss.item())
                step = int((float(epoch) + (batch_idx / len(dataloaders[phase]))) * 1000)
                writer.add_scalars('loss', {phase: loss.item()}, step)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    tempName = idString + str(epoch) + ".pt"
    torch.save(model.state_dict(), tempName)