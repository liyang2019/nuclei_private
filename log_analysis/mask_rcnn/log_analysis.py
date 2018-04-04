import matplotlib.pyplot as plt
import numpy as np

# log_file = 'log_gray500.txt'
# log_file = 'log_purple108.txt'
log_file = '2018-4-1_log_gray500.txt'

valid_losses = np.zeros((0, 6), dtype=np.float)
train_losses = np.zeros((0, 6), dtype=np.float)
with open(log_file, 'r') as f:
    for line in f.readlines():
        line_split = line.split('|')
        if len(line_split) == 5 and line_split[0][:5] != ' rate':

            valid_losses_split = line_split[2].split()
            valid_losses_split = np.array([float(loss.strip()) for loss in valid_losses_split])
            valid_losses = np.vstack([valid_losses, valid_losses_split])

            train_losses_split = line_split[2].split()
            train_losses_split = np.array([float(loss.strip()) for loss in train_losses_split])
            train_losses = np.vstack([train_losses, train_losses_split])


average = 1
filter = np.ones((average,)) / average
for i in range(len(valid_losses)):
    valid_losses[i] = np.convolve(valid_losses[i], filter, mode='same')
    train_losses[i] = np.convolve(train_losses[i], filter, mode='same')
#
# pad = average
# losses = np.array(losses)[:, pad: len(losses[0]) - pad].T
plt.plot(train_losses, linewidth=1)
plt.savefig(log_file.replace('.txt', '.png'))
print(valid_losses.shape)