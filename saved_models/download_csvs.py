import os
import cv2
import pandas
import numpy as np
from mask_rcnn.net.metric import run_length_decode
from mask_rcnn.dataset.reader import *
from test_augmentation import *


if 0:
    # download ensemble csv from gcp
    ensemble_folder = 'ensemble_00037500/none/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/a020_2018-4-15_models_gray690_00037500_model/2018-04-15_20-08-20_ensemble_00037500_model/none/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500/hflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/a020_2018-4-15_models_gray690_00037500_model/2018-04-15_21-25-32_ensemble_00037500_model/hflip/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500/vflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/a020_2018-4-15_models_gray690_00037500_model/2018-04-15_22-38-00_ensemble_00037500_model/vflip/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500/scaleup/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/a020_2018-4-15_models_gray690_00037500_model/2018-04-15_23-50-44_ensemble_00037500_model/scaleup/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500/scaledown/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/a020_2018-4-15_models_gray690_00037500_model/2018-04-16_01-32-39_ensemble_00037500_model/scaledown/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

if 0:
    # download ensemble csv from gcp
    ensemble_folder = 'ensemble_00037500_on_blue/none/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-29-56_ensemble_00037500_model/none/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500_on_blue/hflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-31-10_ensemble_00037500_model/hflip/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500_on_blue/vflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-55-47_ensemble_00037500_model/vflip/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500_on_blue/scaleup/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-32-29_ensemble_00037500_model/scaleup/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500_on_blue/scaledown/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-34-02_ensemble_00037500_model/scaledown/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00037500_on_blue/blur/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/li/nuclei_private/saved_models/2018-4-15_models_gray690_00037500_model_on_blue/2018-04-16_04-34-58_ensemble_00037500_model/blur/submit'
    os.system('gcloud compute scp --recurse --zone us-central1-c instance-1:' + remote_folder + '/submit.csv ' + ensemble_folder)

if 0:
    # download ensemble csv from aws
    ensemble_folder = 'ensemble_00026500/none/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model.pth/2018-04-15_20-41-17_ensemble_00026500_model/none/submit'
    os.system('scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500/hflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model.pth/2018-04-15_22-01-30_ensemble_00026500_model/hflip/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500/vflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model.pth/2018-04-15_23-21-56_ensemble_00026500_model/vflip/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500/scaleup/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model.pth/2018-04-16_00-41-20_ensemble_00026500_model/scaleup/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500/scaledown/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model.pth/2018-04-16_02-30-16_ensemble_00026500_model/scaledown/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

if 0:
    # download ensemble csv from aws
    ensemble_folder = 'ensemble_00026500_on_blue/none/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-51-30_ensemble_00026500_model/none/submit'
    os.system('scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500_on_blue/hflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-52-45_ensemble_00026500_model/hflip/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500_on_blue/vflip/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-54-04_ensemble_00026500_model/vflip/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500_on_blue/scaleup/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-55-20_ensemble_00026500_model/scaleup/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500_on_blue/scaledown/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-56-57_ensemble_00026500_model/scaledown/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

    ensemble_folder = 'ensemble_00026500_on_blue/blur/'
    os.makedirs(ensemble_folder, exist_ok=True)
    remote_folder = '/home/ubuntu/nuclei_private/saved_models/2018-4-15_models_gray800_00026500_model_on_blue/2018-04-16_06-57-54_ensemble_00026500_model/blur/submit'
    os.system(
        'scp -i ~/.ssh/mykey.pem -r ubuntu@ec2-34-217-126-197.us-west-2.compute.amazonaws.com:' + remote_folder + '/submit.csv ' + ensemble_folder)

WRONG_GRAY_KEY = '170bc41b2095177cccd3d4c8977c619147580f1d93b4fe9701eddd77736d4ece'


def name_rle_pair_from_csv(csv_file_name, names):
    df = pandas.read_csv(csv_file_name)
    name_to_rle = {}
    for name in names:
        name_to_rle[name] = []
    for index, row in df.iterrows():
        name = row['ImageId']
        if name == WRONG_GRAY_KEY:
            continue
        name_to_rle[name].append(row['EncodedPixels'])
    return name_to_rle


def csv_to_overlay(file, overlays_folder):
    df = pandas.read_csv(file)
    df.sort_values('ImageId')
    curr_name = None
    curr_mask = None
    count = 0
    im = None
    H, W = None, None
    names = set()

    os.makedirs(overlays_folder, exist_ok=True)
    for index, row in df.iterrows():
        name = row['ImageId']
        if name == WRONG_GRAY_KEY:
            print('what?????????///??????')
            with open('what.txt', 'w') as f:
                print('nooooooo', file=f)
            continue
        names.add(name)
        if curr_name is None or curr_name != name:
            if curr_mask is not None:
                print(name)
                contour_overlay = multi_mask_to_contour_overlay(curr_mask, im, color=[0, 255, 0])
                color_overlay = multi_mask_to_color_overlay(curr_mask, color='summer')
                color1_overlay = multi_mask_to_contour_overlay(curr_mask, color_overlay, color=[255, 255, 255])
                all = np.hstack((im, contour_overlay, color1_overlay))
                cv2.imwrite(overlays_folder + '/%s.png' % curr_name, all)

            im = cv2.imread('../data/2018-4-12_dataset/stage2_test/' + name + '.png')
            H, W = im.shape[0], im.shape[1]
            curr_name = name
            curr_mask = np.zeros((H, W), dtype=np.int)
            count = 0
        rle = row['EncodedPixels']
        mask = run_length_decode(rle, H, W) > 128
        curr_mask[mask] = count + 1
        count += 1

    contour_overlay = multi_mask_to_contour_overlay(curr_mask, im, color=[0, 255, 0])
    color_overlay = multi_mask_to_color_overlay(curr_mask, color='summer')
    color1_overlay = multi_mask_to_contour_overlay(curr_mask, color_overlay, color=[255, 255, 255])
    all = np.hstack((im, contour_overlay, color1_overlay))
    cv2.imwrite(overlays_folder + '/%s.png' % curr_name, all)

    print(len(names))


if __name__ == '__main__':
    print('start ensembling')
    # gray_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/test2_gray_key.txt')]
    # csv_files = [
    #     'ensemble_00026500/none/submit.csv',
    #     'ensemble_00026500/vflip/submit.csv',
    #     'ensemble_00026500/hflip/submit.csv',
    #     'ensemble_00026500/scaleup/submit.csv',
    #     'ensemble_00026500/scaledown/submit.csv',
    #     'ensemble_00037500/none/submit.csv',
    #     'ensemble_00037500/vflip/submit.csv',
    #     'ensemble_00037500/hflip/submit.csv',
    #     'ensemble_00037500/scaleup/submit.csv',
    #     'ensemble_00037500/scaledown/submit.csv',
    # ]
    # name_to_rles_list = []
    # for csv_file in csv_files:
    #     name_to_rles_list.append(name_rle_pair_from_csv(csv_file, gray_keys))
    #
    # csv_file = 'ensemble_total/ensemble_submit.csv'
    # run_ensemble_from_csvs(data_dir='../data/2018-4-12_dataset',
    #                        out_dir='ensemble_total',
    #                        csv_file=csv_file,
    #                        names=gray_keys,
    #                        name_to_rles_list=name_to_rles_list)

    gray_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/test2_gray_key.txt')]
    csv_files = [
        'ensemble_00026500/none/submit.csv',
        'ensemble_00026500/vflip/submit.csv',
        'ensemble_00026500/hflip/submit.csv',
        'ensemble_00026500/scaleup/submit.csv',
        'ensemble_00026500/scaledown/submit.csv',
    ]
    name_to_rles_list = []
    for csv_file in csv_files:
        name_to_rles_list.append(name_rle_pair_from_csv(csv_file, gray_keys))

    out_dir = 'ensemble_total_fewer'
    csv_file = os.path.join(out_dir, 'ensemble_submit.csv')
    run_ensemble_from_csvs(data_dir='../data/2018-4-12_dataset',
                           out_dir=out_dir,
                           csv_file=csv_file,
                           names=gray_keys,
                           name_to_rles_list=name_to_rles_list)

    # blue_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_blue')]
    # csv_files = [
    #     'ensemble_00026500_on_blue/none/submit.csv',
    #     'ensemble_00026500_on_blue/vflip/submit.csv',
    #     'ensemble_00026500_on_blue/hflip/submit.csv',
    #     'ensemble_00026500_on_blue/scaleup/submit.csv',
    #     # 'ensemble_00026500_on_blue/scaledown/submit.csv',
    #     'ensemble_00037500_on_blue/none/submit.csv',
    #     'ensemble_00037500_on_blue/vflip/submit.csv',
    #     'ensemble_00037500_on_blue/hflip/submit.csv',
    #     'ensemble_00037500_on_blue/scaleup/submit.csv',
    #     # 'ensemble_00037500_on_blue/scaledown/submit.csv',
    # ]
    # name_to_rles_list = []
    # for csv_file in csv_files:
    #     name_to_rles_list.append(name_rle_pair_from_csv(csv_file, blue_keys))
    #
    # csv_file = 'ensemble_total_on_blue/ensemble_submit.csv'
    # run_ensemble_from_csvs(data_dir='../data/2018-4-12_dataset',
    #                        out_dir='ensemble_total_on_blu2e',
    #                        csv_file=csv_file,
    #                        names=blue_keys,
    #                        name_to_rles_list=name_to_rles_list)

    # blue_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_blue')]
    # csv_files = [
    #     # 'ensemble_00026500_on_blue/none/submit.csv',
    #     # 'ensemble_00026500_on_blue/vflip/submit.csv',
    #     # 'ensemble_00026500_on_blue/hflip/submit.csv',
    #     # 'ensemble_00026500_on_blue/scaleup/submit.csv',
    #     # 'ensemble_00026500_on_blue/scaledown/submit.csv',
    #     'ensemble_00037500_on_blue/none/submit.csv',
    #     'ensemble_00037500_on_blue/vflip/submit.csv',
    #     'ensemble_00037500_on_blue/hflip/submit.csv',
    #     'ensemble_00037500_on_blue/scaleup/submit.csv',
    #     # 'ensemble_00037500_on_blue/scaledown/submit.csv',
    # ]
    # name_to_rles_list = []
    # for csv_file in csv_files:
    #     name_to_rles_list.append(name_rle_pair_from_csv(csv_file, blue_keys))
    #
    # out_dir = 'ensemble_total_on_blue_fewer'
    # csv_file = os.path.join(out_dir, 'ensemble_submit.csv')
    # run_ensemble_from_csvs(data_dir='../data/2018-4-12_dataset',
    #                        out_dir=out_dir,
    #                        csv_file=csv_file,
    #                        names=blue_keys,
    #                        name_to_rles_list=name_to_rles_list)
