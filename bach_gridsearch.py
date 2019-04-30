import subprocess
from random import choice, randint


def call_main(learning_rate, learning_gamma, hidden_size, number_hidden, frame_size, dropout, number_epochs=20):
    subprocess.call([
        'python', 'main.py',
        '-lr', str(learning_rate),
        '-g', str(learning_gamma),
        '-hs', str(hidden_size),
        '-nh', str(number_hidden),
        '-fs', str(frame_size),
        '-do', str(dropout),
        '-ne', str(number_epochs)
    ])


while True:
    config = {
        'learning_rate': choice([0.001]),
        'learning_gamma': choice([0.8, 0.9]),
        'hidden_size': randint(500, 1500),
        'number_hidden': choice([2]),
        'frame_size': randint(16, 32),
        'dropout': choice([0.5]),
        'number_epochs': 35
    }

    call_main(**config)
