#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach_data import BachDataset
from bach_net import AnalysisNet3
from bach_Synthesizer import Synthesizer
from bach_datadownloader_new import DataDownloader2
from bach_neurons_function import NeuronsFunctionComparator

# hyperparameters

hiddenSize = 1024
numberHidden = 2
frameSize = 32
dropout = 0.5

batchSize = 1

data_path = os.path.join('.', 'chordDataSQ')
if not os.path.exists(data_path):
    downloader = DataDownloader2(data_path, [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], True)
    downloader.download()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': batchSize, 'shuffle': True}  # 'num_workers': 2}    #doesnt work: file open in getitem? google

# Datasets
dataloaders = {
    'train': DataLoader(BachDataset(os.path.join(data_path, 'train'), frameSize, batchSize), **params)#,
    #'valid': DataLoader(BachDataset(os.path.join(data_path, 'valid'), frameSize, batchSize), **params)
}

model = AnalysisNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden, dropout=dropout).to(device)  # former hidden 600 - wasnt better?

model.load_state_dict(torch.load('04-16 10-51-lr0.001-g0.9-hs1024-nh2-fs16-do0.5-24.pt', map_location='cpu'))
model.eval()

nfc = NeuronsFunctionComparator()

for phase in ['train']:#, 'valid']:

    for batch_idx, batch_sample in enumerate(tqdm(dataloaders[phase], desc=phase, unit='batch')):
        batch, labels = batch_sample
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.permute(2, 0, 1)
        labels = labels.permute(2, 0, 1)

        s = Synthesizer()
        testOut = labels.detach().numpy()[:,0,:]
        batchOut = batch.detach().numpy()[:,0,:]
        testOut[:,:7] = batchOut[:,:7]
        testOut[:,194:] = batchOut[:,194:]
        score = s.synthesizeFromArray(labels.detach().numpy()[:,0,:], False)

        with torch.set_grad_enabled(phase != 'train'):
            h = model.init_hidden(len(batch[0]))
            y_pred, hidden, _ = model(batch, h)
            nfc.compare(score, _, "minormajor")


