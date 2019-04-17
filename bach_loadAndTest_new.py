import torch
from bach_net import AnalysisNet3, BachNet3
import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note, converter, metadata
from bach_Synthesizer import Synthesizer
from bach_partToDataArray import PartConverter

D_in, H, D_out = 278, 1279, 278
model = BachNet3(D_in, H, D_out, 3)

# gut:
# model.load_state_dict(torch.load('/Users/alexanderleemhuis/Informatik/PY/PRJ/2019_04_01 bach5/MODEL decaying lr 7.4./model16.pt'))
#model.load_state_dict(torch.load('/Users/alexanderleemhuis/Informatik/PY/PRJ/2019_04_01 bach5/MODEL 20epo/model19.pt', map_location='cpu'))
model.load_state_dict(torch.load('04-16 14-35-lr0.001-g0.9-hs1279-nh3-fs16-do0.5-19.pt', map_location='cpu'))
model.eval()

pc = PartConverter()
data = converter.parse(
    './xml test/38.xml')#.transpose(5)
    #'/Users/alexanderleemhuis/Informatik/PY/PRJ/2019_04_01 bach5/xml test/mond-A-noks.musicxml')#.transpose(-1)
dataI, _, _, _ = pc.convertToDataArray(data, "piece", True)
np.savetxt("debug input test.csv", dataI, fmt='%d')
dataI = torch.unsqueeze(torch.tensor(dataI), 1)
dataI = dataI.float()
dataI = np.swapaxes(dataI, 0, 2)

h = model.init_hidden(len(dataI[0]))
y_pred, hidden = model(torch.tensor(dataI), h)
y_np = y_pred.detach().numpy()

print(dataI.shape)  # length, batch=1, 278vals
print(y_np.shape)

y_np[:,0,:7] = dataI[:,0,:7]
y_np[:,0,194:] = dataI[:,0,194:]

y_np = np.squeeze(y_np, axis=1)

bachSynth = Synthesizer()
bachSynth.synthesizeFromArray(y_np)
