#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from bach_data_for_analysis import BachDataset
from bach_net import AnalysisNet3, PlotNeuronNet3
from bach_Synthesizer import Synthesizer
from bach_datadownloader_new import DataDownloader

score_and_neurons_path = os.path.join('.', 'neuron_analysis_results')
data_path = os.path.join('.', 'data_copies_for_analysis')

hiddenSize = 1103
numberHidden = 2
dropout = 0.5
batchSize = 1

if os.path.exists(score_and_neurons_path):
    print("Analysis folder found. Quitting. To run analysis again, delete folder.")
    raise SystemExit

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

model = AnalysisNet3(input_dims=279, hidden_dims=hiddenSize, output_dims=279, num_hidden_layers=numberHidden, dropout=dropout, neurons=plot_pairs).to(device)  # former hidden 600 - wasnt better?
model.load_state_dict(torch.load('04-30 12-03-lr0.001-g0.8-hs1103-nh2-fs18-do0.5-30.pt', map_location='cpu'))

model.eval()

epo_stop = 10
counter = 0

for phase in ['train']:  #, 'valid']:

    for batch_idx, batch_sample in enumerate(tqdm(dataloaders[phase], desc=phase, unit='batch')):
        batch, labels, current_title = batch_sample
        current_title = current_title[0]
        print(current_title)
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.permute(2, 0, 1)
        labels = labels.permute(2, 0, 1)

        with torch.set_grad_enabled(phase != 'train'):
            h = model.init_hidden(len(batch[0]))
            y_pred, hidden, neurons = model(batch, h)

            s = Synthesizer()
            netOut = y_pred.detach().numpy()[:,0,:]
            batchOut = batch.detach().numpy()[:,0,:]
            netOut[:,:7] = batchOut[:,:7]
            netOut[:,194:] = batchOut[:,194:]

            show_score = False
            score = s.synthesizeFromArray(netOut, show_score, False)   # no ties to prepare analysis!


            path = os.path.join(score_and_neurons_path, current_title)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            for i, neuron in enumerate(neurons):
                file_name = "layer" + str(i+1) + ".csv"
                path_neuron = os.path.join(path, file_name)
                np.savetxt((path_neuron), neuron)

            score_name = current_title + ".musicxml"
            path_score = os.path.join(path, score_name)
            score.write("musicxml", path_score)

        if counter >= epo_stop:
            break

        counter += 1



