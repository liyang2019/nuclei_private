import os

# download models from AWS

# ssh_address = 'ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com'
# awskey_loc = '~/.ssh/mykey.pem'
#
# models_gray800_folder = 'models_gray800'
# os.makedirs(models_gray800_folder, exist_ok=True)
#
# os.system('scp -i %s -r %s:/home/ubuntu/nuclei_private/results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00018000_model.pth %s' % (awskey_loc, ssh_address, models_gray800_folder))
# os.system('scp -i %s -r %s:/home/ubuntu/nuclei_private/results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00018500_model.pth %s' % (awskey_loc, ssh_address, models_gray800_folder))
# os.system('scp -i %s -r %s:/home/ubuntu/nuclei_private/results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00019000_model.pth %s' % (awskey_loc, ssh_address, models_gray800_folder))
#

# download models from gcp

# 35000 # 0.504
# 32500 # 0.514 is this the best?
# 31500 # 0.504
# 29000 # 0.507
# 27500 # 0.503

models_gray690_folder = 'models_gray690'
os.makedirs(models_gray690_folder, exist_ok=True)
IDENTIFIER = '2018-04-14_03-14-43'

os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-14_03-14-43/checkpoint/00035000_model.pth %s' % models_gray690_folder)
os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-14_03-14-43/checkpoint/00032500_model.pth %s' % models_gray690_folder)
os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-14_03-14-43/checkpoint/00031500_model.pth %s' % models_gray690_folder)
os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-14_03-14-43/checkpoint/00029000_model.pth %s' % models_gray690_folder)
os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:/home/li/nuclei_private/results/2018-4-13_gray690_full/2018-04-14_03-14-43/checkpoint/00027500_model.pth %s' % models_gray690_folder)

