import torch

import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note, converter, metadata

from model import BachNet
from synthesizer import Synthesizer
from utils.part_to_data_array import PartConverter

loaded = torch.load('05-23 12-12-lr0.005-g0.98-ls5-hs45-nh2-fs32-do0.5-222.pt', map_location='cpu')
config = loaded['config']

D_in, H, D_out = 279, config['hidden_size'], 62
model = BachNet(D_in, H, D_out, 2)

# gut:
# model.load_state_dict(torch.load('04-30 14-46-lr0.001-g0.9-hs1002-nh2-fs21-do0.5-22.pt', map_location='cpu'))
model.load_state_dict(loaded['state'])
model.eval()

pc = PartConverter()
data = converter.parse(
#    './xml test/floskel.musicxml')
#    './xml test/38 long.musicxml')
#    './xml test/211 schluss.musicxml')   # Kaffeekantate
#    './xml test/05.musicxml')  # bassdurchgänge
#    './xml test/06 sopran.musicxml')   # ausweichungen
    './xml test/kirby fsharp.mxl')


# carefull: no repeats are extended in test-data!
dataI, _, _, _ = pc.convertToDataArray(data, "piece", True)
np.savetxt("debug input test.csv", dataI, fmt='%d')
dataI = torch.unsqueeze(torch.tensor(dataI), 1)
dataI = dataI.float()
dataI = np.swapaxes(dataI, 0, 2)

timestep_number = dataI[:, 0, 0].shape[0]

h = model.init_hidden(len(dataI[0]))
y_pred_B, y_pred_A, y_pred_T, hidden = model(torch.tensor(dataI), h)

current_batchsize = y_pred_A[0, :, 0].shape[0]
beforepad = torch.zeros(timestep_number, current_batchsize, 8)  # padding for format
afterpad = torch.zeros(timestep_number, current_batchsize, 85)  # padding for format
y_pred = torch.cat((beforepad, y_pred_B, y_pred_T, y_pred_A, afterpad), 2)


y_np = y_pred.detach().numpy()

print(dataI.shape)  # length, batch=1, 278vals
print(y_np.shape)

y_np[:,0,:7] = dataI[:,0,:7]
y_np[:,0,194:] = dataI[:,0,194:]

y_np = np.squeeze(y_np, axis=1)

bachSynth = Synthesizer()
bachSynth.synthesizeFromArray(y_np)
