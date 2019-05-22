import torch

import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note, converter, metadata

from .model import BachNet
from .synthesizer import Synthesizer
from .utils.part_to_data_array import PartConverter

D_in, H, D_out = 279, 1002, 279
model = BachNet(D_in, H, D_out, 2)

# gut:
model.load_state_dict(torch.load('04-30 14-46-lr0.001-g0.9-hs1002-nh2-fs21-do0.5-22.pt', map_location='cpu'))
model.eval()

pc = PartConverter()
data = converter.parse(
#    './xml test/Alex AI fermate.musicxml')# .transpose(5)
    './xml test/Auferstanden_Aus_Ruinen.musicxml')# .transpose(5)
#   './xml test/38.xml')# .transpose(5)
#    './xml test/test1ks.musicxml')# .transpose(5)
#    './xml test/mond-A-Aks.musicxml')# .transpose(5)


# carefull: no repeats are extended in test-data!
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
