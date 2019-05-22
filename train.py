from random import choice, randint
from tqdm import tqdm

from trainer import train

config = {
    'learning_rate': 0.001,
    'learning_gamma': 0.9,
    'learning_step': 3,
    'hidden_size': 1115,
    'number_hidden': 2,
    'frame_size': 16,
    'dropout': 0.5,
    'number_epochs': 100
}

train(config)