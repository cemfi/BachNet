import math
from random import randrange
from glob import glob

from torch.utils.data import Dataset
import numpy as np
import os


class BachDataset(Dataset):
    def __init__(self, root_dir, frame_size):
        self.input = []
        self.output = []

        for directory in glob(os.path.join(root_dir, '*')):
            cur_input = np.load(os.path.join(directory, 'i.npy'))
            cur_output = np.load(os.path.join(directory, 'o.npy'))

            num_available_steps = math.ceil(cur_input.shape[1] / frame_size)

            for step_idx in range(num_available_steps):
                if len(cur_input[0, :]) < frame_size:
                    raise Exception("File smaller than frame_size")

                if (step_idx + 1) * frame_size <= len(cur_input[0, :]):
                    start = step_idx * frame_size
                else:
                    start = len(cur_input[0, :]) - frame_size

                self.input.append(cur_input[:, start:start + frame_size])
                self.output.append(cur_output[:, start:start + frame_size])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        index = randrange(0, len(self))
        return self.input[index], self.output[index]
