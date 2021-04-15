import re
from glob import glob

import matplotlib.pyplot as plt

loss_per_epoch_bass = {}
loss_per_epoch_middle = {}
expr = re.compile(r'bass=(.+) middle=(.+) e=(\d+)')

file_paths = glob('./all_test_pieces_plain/**/*.midi')

for fp in file_paths:
    bass, middle, epoch = expr.findall(fp)[0]
    if epoch not in loss_per_epoch_bass:
        loss_per_epoch_bass[epoch] = 0
        loss_per_epoch_middle[epoch] = 0
    loss_per_epoch_bass[epoch] += float(bass)
    loss_per_epoch_middle[epoch] += float(middle)

# Bass
sorted_loss_bass = [(int(k), loss_per_epoch_bass[k]) for k in
                    sorted(loss_per_epoch_bass, key=loss_per_epoch_bass.get, reverse=True)]
print('Part part:')
for e, l in sorted_loss_bass:
    print(e, l)

sorted_loss_bass_epochs = sorted(loss_per_epoch_bass)

epochs_bass = [int(e) for e in sorted_loss_bass_epochs]
losses_bass = [loss_per_epoch_bass[e] for e in sorted_loss_bass_epochs]
plt.plot(epochs_bass, losses_bass, label='bass')

print()

# Middle
sorted_loss_middle = [(int(k), loss_per_epoch_middle[k]) for k in
                      sorted(loss_per_epoch_middle, key=loss_per_epoch_middle.get, reverse=True)]
print('Middle parts:')
for e, l in sorted_loss_middle:
    print(e, l)

sorted_loss_middle_epochs = sorted(loss_per_epoch_middle)

epochs_middle = [int(e) for e in sorted_loss_middle_epochs]
losses_middle = [loss_per_epoch_middle[e] for e in sorted_loss_middle_epochs]
plt.plot(epochs_middle, losses_middle, label='middle')

# Both
losses_all = []
for i in range(len(losses_bass)):
    losses_all.append(losses_bass[i] + losses_middle[i])
plt.plot(epochs_bass, losses_all, label='all')
plt.legend(loc='upper left')

plt.show()
