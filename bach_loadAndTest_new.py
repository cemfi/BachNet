import torch
from bach_net import AnalysisNet3, BachNet3
import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note, converter, metadata
from bach_Synthesizer import Synthesizer
from bach_partToDataArray import PartConverter


D_in, H, D_out = 279, 1103, 279
#D_in, H, D_out = 279, 1115, 279


model = BachNet3(D_in, H, D_out, 2)

#model.load_state_dict(torch.load('05-06 14-44-lr0.003-g0.99-hs35-nh2-fs16-do0.5-500.pt', map_location='cpu'))

model.load_state_dict(torch.load('04-30 12-03-lr0.001-g0.8-hs1103-nh2-fs18-do0.5-30 gut.pt', map_location='cpu'))
#model.load_state_dict(torch.load('04-30 12-46-lr0.001-g0.9-hs1115-nh2-fs16-do0.5-19 gut bei kirchen.pt', map_location='cpu'))
#model.load_state_dict(torch.load('04-30 12-46-lr0.001-g0.9-hs1115-nh2-fs16-do0.5-99 over.pt', map_location='cpu'))

model.eval()

pc = PartConverter()
data = converter.parse(
#    './xml test/38 long.musicxml')
#    './xml test/05.musicxml')  # bassdurchgänge
#    './xml test/06 sopran.musicxml')   # ausweichungen
    './xml test/kirby fsharp.mxl')
#    './xml test/mond-A-Aks.musicxml')

#    './xml test/211 schluss.musicxml')   # Kaffeekantate

dataI, _, _, _ = pc.convertToDataArray(data, "piece", True)
np.savetxt("debug input test.csv", dataI, fmt='%d')
dataI = torch.unsqueeze(torch.tensor(dataI), 1)
dataI = dataI.float()
dataI = np.swapaxes(dataI, 0, 2)

h = model.init_hidden(len(dataI[0]))
y_pred, hidden = model(torch.tensor(dataI), h)
y_np = y_pred.detach().numpy()

print(dataI.shape)  # length, batch=1, 279 vals
print(y_np.shape)

y_np[:,0,:7] = dataI[:,0,:7]
y_np[:,0,194:] = dataI[:,0,194:]

y_np = np.squeeze(y_np, axis=1)

bachSynth = Synthesizer()
score = bachSynth.synthesizeFromArray(y_np,piano_reduction=True)
print(score.write('lily'))
