import os
from log_analysis.mask_rcnn.log_analysis import analyze_log
import matplotlib.pyplot as plt


result_folder = '2018-4-14_gray800_box'
# IDENTIFIER = '2018-04-14_18-16-22'
IDENTIFIER = '2018-04-15_03-47-56'
result_dir = os.path.join(result_folder, IDENTIFIER)
if 1:
    os.makedirs(result_dir + '/train', exist_ok=True)
    os.system('scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:/home/ubuntu/nuclei_private/results/' + result_dir + '/train/* ' + result_dir + '/train')
    os.system('scp -i ~/.ssh/mykey.pem ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:/home/ubuntu/nuclei_private/mask_rcnn/log.txt ' + result_dir)
    os.system('scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:/home/ubuntu/nuclei_private/results/' + result_folder + '/latest_model.pth ' + result_dir + '/')

    print('success')


valid_losses, train_losses, valid_acc = analyze_log(result_dir + '/log.txt', 1)

start = 1488
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

plt.figure()
plt.plot(valid_acc[start:, 0], '.b-', linewidth=0.5, markersize=1)
plt.savefig(result_folder + '_' + IDENTIFIER + '_metric.png', dpi=300)
