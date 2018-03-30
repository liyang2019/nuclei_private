import matplotlib.pyplot as plt
import numpy as np
# log_file = 'log_wxwang_purple108.txt'
log_file = 'log_ly_purple108_augmentation.txt'

steps = []
loss = []
with open(log_file, 'r') as f:
    for line in f.readlines():
        if line[:5] == 'epoch':
            line_split = line.split('|')
            steps.append(int(line_split[1][6:].strip()))
            loss.append(float(line_split[2][11:].strip()))
average = 100
loss = np.convolve(loss, np.ones((average,))/average, mode='same')
plt.figure()
pad = 50
plt.plot(steps[pad:len(steps) - pad], loss[pad:len(steps) - pad], linewidth=1)
plt.savefig(log_file.replace('.txt', '.png'))
print(len(loss))