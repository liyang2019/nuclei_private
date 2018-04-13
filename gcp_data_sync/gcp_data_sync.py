import os

result_folder = '2018-4-13_gray690_box'
os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/' + result_folder + ' ./')

os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/mask_rcnn/log.txt ' + result_folder + '/')