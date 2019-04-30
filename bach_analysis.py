#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach_data_for_analysis import BachDataset
from bach_net import AnalysisNet3, PlotNeuronNet3
from bach_Synthesizer import Synthesizer
from bach_datadownloader_new import DataDownloader
from bach_neurons_function import NeuronsFunctionComparator

# hyperparameters

analysis_neurons = [1797, 1961, 2160,2162] #  None # [1898, 1942, 2106]


if analysis_neurons is None:
    analysis_mode = True
    showScore = False  # show if anaylsis mode
else:
    analysis_mode = False
    showScore = True

# else quasi plotmode

hiddenSize = 1165
numberHidden = 2
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
    'train': DataLoader(BachDataset(os.path.join(data_path, 'train')), **params)
}

if analysis_mode:
    model = AnalysisNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden, dropout=dropout).to(device)  # former hidden 600 - wasnt better?
    model.load_state_dict(torch.load('04-18 09-46-lr0.001-g0.9-hs1165-nh2-fs26-do0.5-49.pt', map_location='cpu'))

else:
    models = []
    for neuron in analysis_neurons:
        model = PlotNeuronNet3(input_dims=278, hidden_dims=hiddenSize, output_dims=278, num_hidden_layers=numberHidden,
                         dropout=dropout, neuron=neuron).to(device)
        model.load_state_dict(torch.load('04-18 09-46-lr0.001-g0.9-hs1165-nh2-fs26-do0.5-49.pt', map_location='cpu'))
        models.append(model)

model.eval()

nfc = NeuronsFunctionComparator(hiddenSize * 2)

for phase in ['train']:#, 'valid']:

    for batch_idx, batch_sample in enumerate(tqdm(dataloaders[phase], desc=phase, unit='batch')):
        batch, labels = batch_sample
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.permute(2, 0, 1)
        labels = labels.permute(2, 0, 1)


        with torch.set_grad_enabled(phase != 'train'):
            if analysis_mode:
                h = model.init_hidden(len(batch[0]))
                y_pred, hidden, neurons = model(batch, h)
            else:
                for model in models:
                    h = model.init_hidden(len(batch[0]))
                    y_pred, hidden = model(batch, h)

            s = Synthesizer()
            netOut = y_pred.detach().numpy()[:,0,:]
            batchOut = batch.detach().numpy()[:,0,:]
            netOut[:,:7] = batchOut[:,:7]
            netOut[:,194:] = batchOut[:,194:]

            score = s.synthesizeFromArray(netOut, showScore)

            maxChords = len(netOut[:,1])
            template = nfc.analyze(score, maxChords)
            if showScore:
                plt.plot(template)
            plt.show()

            nfc.store_neurons(neurons)


nfc.find_correlations()


