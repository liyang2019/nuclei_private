import os

from net.metric import run_length_decode

os.system('export PYTHONPATH=../:../mask_rcnn/:../ensemble')
os.system('echo $PYTHONPATH')

import pandas as pd
import argparse
import cv2
from saved_models.download_csvs import csv_to_overlay
from mask_rcnn.dataset.reader import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run segmentation models')
    parser.add_argument('--csv_file', action='store', default='final_combined.csv')
    parser.add_argument('--overlays_folder', action='store', default='debug_folder')
    args = parser.parse_args()

    gray_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/test2_gray_key.txt')]
    blue_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_blue')]
    purple_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_purple')]
    external_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_external')]

    print(len(gray_keys))
    print(len(blue_keys))
    print(len(purple_keys))
    print(len(external_keys))

    # csv_file = 'gray_blue_combined.csv'
    csv_file = args.csv_file
    df = pd.read_csv(csv_file)

    # names = set()
    #
    # for index, row in df.iterrows():
    #     names.add(row['ImageId'])
    #     if row['EncodedPixels'] == '':
    #         print(row['ImageId'])
    #
    # print(len(df))
    # print(names == set(gray_keys + blue_keys + purple_keys + external_keys))
    #
    # # csv_to_overlay(csv_file, 'overlays')
    # csv_to_overlay(csv_file, args.overlays_folder)

    # check 2
    name_to_rles = {}
    names = gray_keys + blue_keys + purple_keys + external_keys
    assert len(names) == 3019
    for name in names:
        name_to_rles[name] = []

    for index, row in df.iterrows():
        name_to_rles[row['ImageId']].append(row['EncodedPixels'])

    # name_sample = [names[1], names[323], names[433], names[2232], names[315]]
    name_sample = [
        '0a5a390c955100df18bc11ba62721756487573a47ea5a5ea3f3ebc01d9789794',
        '0aa80c97331a51eab952724ae16840d2d36632d0dd7e578afe707743a4854332',
        '0abb58cc7ac43623febe10542a2c0d5230652cb092bf63ff4dc69adb5de0cd3d',
        '0ca87beee0808d4865973ee05aeaac803e836984bc6d64796c4508d094ee6cb6',
        '0eb4feb7ab8c3c6c129dd0abda6a7fec8c477d933484b4581a20dca249d1b12d',
        '0fb2554bc5ba639190ed1121fe14a7fd82323c378decc1fdd9f8fe696d8478f1',
        '1c4e26959e8326d80104ca7c1f9f21924646aceee86242d6daba338a59154e33',
        '2fac642d6058508156612457efc5008fe266c1b0c8f7c8b15b79ead078e91ef3',
        '3f3ef6449fe24832d61021365eabfb6ce21fbb17c63de03714e7dea2941193b2',
        '5d57198d730b986ac6a249b4ccfebd9fc9ecb44f8d2c165a566a8afb7331be83',
        '5fe25ac2240cd2c985d44fef11b604608ca643ba1cce862b4ded4a05a9f25816',
        '6f152c1c13a1fd513fbe3fa87cc046c95348fd7c39614ff495c2f758b81ece5a',
        '8b80aca225c1dd0aa5637229f0e0bec0dbd7b5313b462effd768da2bb5a56a06',
        '9c1ad3fcb45d89072864a18608a0778468e7580b2b8207ae1cf5d096d380368a',
        '33d6d8e9d74f9da9679000a6cf551fffe4ad45af7d9679e199c5c4bd2d1e0741',
        '39b6ed3e0dd50ea6e113e4f95b656c9bdc6d3866579f648a16a853aeb5af1a61',
        '5390acefd575cf9b33413ddf6cbb9ce137ae07dc04616ba24c7b5fe476c827d2',
        '902528fc53b023c370d6042895151b9e23aae90b22b87c394669ef681f918b84',
        'f22874c8fac3c7c39297796cf8e03f1b5c18a1a2497973ae9338f7f7dd46d36b',

    ]
    for name in name_sample:
        rles = name_to_rles[name]
        print(rles)
        im = cv2.imread('/Users/li/2018_Data_Science_Bowl/nuclei_private/data/stage2_test_final/%s/images/%s.png' % (name, name))
        H, W = im.shape[0], im.shape[1]
        mask = np.zeros((H, W), dtype=np.int)
        for i, rle in enumerate(rles):
            m = run_length_decode(rle, H, W) > 128
            mask[m] = i + 1

        contour_overlay = multi_mask_to_contour_overlay(mask, im, color=[0, 255, 0])
        color_overlay = multi_mask_to_color_overlay(mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay, color=[255, 255, 255])
        all = np.hstack((im, contour_overlay, color1_overlay))
        cv2.imwrite(args.overlays_folder + '/%s.png' % name, all)
