import matplotlib.pyplot as plt
import numpy as np

# log_file = 'log_gray500.txt'
log_file = 'log_purple108.txt'

train_loss = []
rpn_cls_loss = []
rpn_reg_loss = []
rcnn_cls_loss = []
rcnn_reg_loss = []
mask_cls_loss = []
losses = [train_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, mask_cls_loss]
with open(log_file, 'r') as f:
    for line in f.readlines():
        line_split = line.split('|')
        if len(line_split) == 5 and line_split[0][:5] != ' rate':
            train_losses = line_split[2]
            train_losses_split = train_losses.split()
            for i, loss in enumerate(train_losses_split):
                losses[i].append(float(train_losses_split[i].strip()))

average = 100
filter = np.ones((average,)) / average
for i in range(len(losses)):
    losses[i] = np.convolve(losses[i], filter, mode='same')

pad = average
losses = np.array(losses)[:, pad: len(losses[0]) - pad].T
plt.plot(losses, linewidth=1)
plt.savefig(log_file.replace('.txt', '.png'))
print(losses.shape)