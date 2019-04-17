from random import randrange
from glob import glob

from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm


class BachDataset(Dataset):
    def __init__(self, root_dir, frame_size, batch_length):  # dataPrefix = directory
        self.batchLength = frame_size
        self.directories = []
        self.startElements = []

        for directory in tqdm(glob(os.path.join(root_dir, '*'))):
            file_path = os.path.join(directory, 'i.npy')
            data = np.load(file_path)
            batch_remaining = 0 if len(data[0, :]) % frame_size == 0 else 1

            available_steps = data.shape[1] // frame_size + batch_remaining

            for i in range(available_steps):
                self.directories.append(directory)
                if len(data[0, :]) < frame_size:
                    #print(directory)
                    #print(len(data[0, :]))
                    raise Exception("File smaller than frame_size")
                elif (i + 1) * frame_size <= len(data[0, :]):
                    self.startElements.append(i * frame_size)
                else:
                    self.startElements.append(len(data[0, :]) - frame_size)
        #rest = len(self.directories) % 40
        #if rest != 0:
        #    del self.directories[-rest:]
        #    del self.startElements[-rest:]
        #print(self.directories)

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        index = randrange(0, self.__len__())
        directory = self.directories[index]

        input_file_path = os.path.join(directory, 'i.npy')
        input_data = np.load(input_file_path)

        output_file_path = os.path.join(directory, 'o.npy')
        output_data = np.load(output_file_path)

        start_element = self.startElements[index]
        end_element = start_element + self.batchLength

        return input_data[:, start_element:end_element], output_data[:, start_element:end_element]
