import subprocess
from random import choice

from tqdm import tqdm

learningRate = [0.001] #, 0.01, 0.1]   #0.001 einzig wahres
gamma = [0.5, 0.7, 0.9, 1]
hiddenSize = [64, 128, 256, 512]    #64 auch mies
numberHidden = [1, 2, 3, 4, 5]      #5 layers zu viel
frameSize = [16, 32]
dropout =[0.1, 0.2, 0.3, 0.4, 0.5]
comment = ""


def call_main(lr, g, hs, nh, fs, do):
    subprocess.call([
        'python3', 'Main4.py',
        '-lr', str(lr),
        '-g', str(g),
        '-hs', str(hs),
        '-nh', str(nh),
        '-fs', str(fs),
        '-do', str(do)
        ]
    )

if __name__ == '__main__':
    for i in tqdm(range(100)):
        call_main(
            choice(learningRate),
            choice(gamma),
            choice(hiddenSize),
            choice(numberHidden),
            choice(frameSize),
            choice(dropout)
        )
