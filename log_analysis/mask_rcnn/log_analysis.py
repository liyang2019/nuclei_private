import matplotlib.pyplot as plt
import numpy as np


def analyze_log(f):
    log_file = f

    valid_losses = np.zeros((0, 6), dtype=np.float)
    train_losses = np.zeros((0, 6), dtype=np.float)
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line_split = line.split('|')
            if len(line_split) == 5 and line_split[0][:5] != ' rate':
                valid_losses_split = line_split[1].split()
                valid_losses_split = np.array([float(loss.strip()) for loss in valid_losses_split])
                valid_losses = np.vstack([valid_losses, valid_losses_split])

                train_losses_split = line_split[2].split()
                train_losses_split = np.array([float(loss.strip()) for loss in train_losses_split])
                train_losses = np.vstack([train_losses, train_losses_split])

    average = 1
    filter = np.ones((average,)) / average
    for i in range(valid_losses.shape[1]):
        valid_losses[:, i] = np.convolve(valid_losses[:, i], filter, mode='same')
        train_losses[:, i] = np.convolve(train_losses[:, i], filter, mode='same')

    smooth_length = 10
    valid_losses = smooth(valid_losses, smooth_length)
    train_losses = smooth(train_losses, smooth_length)

    # plt.plot(train_losses, 'r', linewidth=1)
    # plt.plot(valid_losses, 'b', linewidth=1)
    # plt.savefig(log_file.replace('.txt', '.png'))
    print(valid_losses.shape)
    return valid_losses, train_losses


def smooth(arr, smooth_length):
    res = np.zeros(arr.shape)
    cum = np.zeros(arr.shape[1])
    for i in range(arr.shape[0]):
        if i < smooth_length:
            cum += arr[i, :]
            res[i, :] = cum / (i + 1)
        else:
            cum += arr[i, :] - arr[i - smooth_length, :]
            res[i, :] = cum / smooth_length
    return res


if __name__ == '__main__':
    valid_losses1, train_losses1 = analyze_log('2018-4-10_mini-unet.txt')
    valid_losses2, train_losses2 = analyze_log('2018-4-10_mini-res.txt')
    valid_losses3, train_losses3 = analyze_log('2018-4-10_4conv.txt')

    plt.figure()
    plt.plot(valid_losses1[:, 5], 'r')
    plt.plot(valid_losses2[:, 5], 'g')
    plt.plot(valid_losses3[:, 5], 'b')
    plt.savefig('2018-4-10_mask_compare_valid.png')

    plt.figure()
    plt.plot(train_losses1[:, 5], 'r')
    plt.plot(train_losses2[:, 5], 'g')
    plt.plot(train_losses3[:, 5], 'b')
    plt.savefig('2018-4-10_mask_compare_train.png')