import os
from log_analysis.mask_rcnn.log_analysis import analyze_log
import matplotlib.pyplot as plt


result_folder = '2018-4-13_gray690_full'
IDENTIFIER = '2018-04-13_14-39-13'
if 1:
    # os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/' + result_folder + '/2018-04-13_14-39-13/train' + ' ./')
    # os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/' + result_folder + '/latest_model.pth' + ' ./')
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-13_14-39-13/checkpoint/00016500_model.pth ./')
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/mask_rcnn/log.txt ./')

    print('success')


valid_losses, train_losses = analyze_log('log.txt', 1)

start = 1200
plt.figure()
plt.plot(valid_losses[start:, 0], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 0], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_tot.png', dpi=300)

plt.figure()
plt.plot(valid_losses[start:, 1], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 1], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_rpn_cls.png', dpi=300)

plt.figure()
plt.plot(valid_losses[start:, 2], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 2], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_rpn_box.png', dpi=300)

plt.figure()
plt.plot(valid_losses[start:, 3], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 3], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_rcnn_cls.png', dpi=300)

plt.figure()
plt.plot(valid_losses[start:, 4], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 4], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_rcnn_box.png', dpi=300)

plt.figure()
plt.plot(valid_losses[start:, 5], '.b-', linewidth=0.5, markersize=1)
plt.plot(train_losses[start:, 5], '.r-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_mask.png', dpi=300)
