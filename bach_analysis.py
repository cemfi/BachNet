#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach_data import BachDataset
from bach_net import AnalysisNet3, PlotNeuronNet3
from bach_Synthesizer import Synthesizer
from bach_datadownloader_new import DataDownloader
from bach_neurons_function import NeuronsFunctionComparator

# hyperparameters

analysis_neurons = [112, 350, 466, 679]  # if none: analyse, if number: visualize neuron
# dominanten: 466, 350, 679

# 336?
# 339
# 374
# 375
# 379
# 409
# 410
# 465
analysis_mode = False

if analysis_neurons is None:
    analysis_mode = True
# else quasi plotmode

hiddenSize = 400
numberHidden = 2
frameSize = 32
dropout = 0.5

batchSize = 1

data_path = os.path.join('.', 'data')
if not os.path.exists(data_path):
    downloader = DataDownloader(data_path, [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], True)
    downloader.download()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': batchSize, 'shuffle': True}  # 'num_workers': 2}    #doesnt work: file open in getitem? google

# Datasets
dataloaders = {
    'train': DataLoader(BachDataset(os.path.join(data_path, 'train'), frameSize), **params)
}

if analysis_mode:
    model = AnalysisNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden, dropout=dropout).to(device)  # former hidden 600 - wasnt better?
    model.load_state_dict(torch.load('04-18 11-42-lr0.001-g0.9-hs400-nh2-fs16-do0.5-35.pt', map_location='cpu'))

else:
    models = []
    for neuron in analysis_neurons:
        model = PlotNeuronNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden,
                         dropout=dropout, neuron=neuron).to(device)
        model.load_state_dict(torch.load('04-18 11-42-lr0.001-g0.9-hs400-nh2-fs16-do0.5-35.pt', map_location='cpu'))
        models.append(model)

model.eval()

nfc = NeuronsFunctionComparator()

for phase in ['train']:#, 'valid']:

    for batch_idx, batch_sample in enumerate(tqdm(dataloaders[phase], desc=phase, unit='batch')):
        batch, labels = batch_sample
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.permute(2, 0, 1)
        labels = labels.permute(2, 0, 1)


        with torch.set_grad_enabled(phase != 'train'):
            if analysis_mode:
                h = model.init_hidden(len(batch[0]))
                y_pred, hidden, _ = model(batch, h)
            else:
                for model in models:
                    h = model.init_hidden(len(batch[0]))
                    y_pred, hidden = model(batch, h)

            s = Synthesizer()
            testOut = y_pred.detach().numpy()[:,0,:]
            batchOut = batch.detach().numpy()[:,0,:]
            testOut[:,:7] = batchOut[:,:7]
            testOut[:,194:] = batchOut[:,194:]
            showScore = not analysis_mode  # show if anaylsis mode
            score = s.synthesizeFromArray(testOut, showScore)

            plt.show()
            nfc.compare(score, _, frameSize, "chord-type")


