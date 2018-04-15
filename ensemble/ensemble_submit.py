from datetime import datetime
import math

import cv2
import pandas as pd

from color_classifier import *
from mask_rcnn.submit import run_submit, run_npy_to_sumbit_csv

ALL_TEST_IMAGE_ID = [
    '0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5',
    '0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac',
    '0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732',
    '0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925',
    '0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6',
    '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466',
    '1747f62148a919c8feb6d607faeebdf504b5e2ad42b6b1710b1189c37ebcdb2c',
    '17b9bf4356db24967c4677b8376ac38f826de73a88b93a8d73a8b452e399cdff',
    '1879f4f4f05e2bada0ffeb46c128b8df7a79b14c84f38c3e216a69653495153b',
    '191b2b2205f2f5cc9da04702c5d422bc249faf8bca1107af792da63cccfba829',
    '1962d0c5faf3e85cda80e0578e0cb7aca50826d781620e5c1c4cc586bc69f81a',
    '1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e',
    '1d9eacb3161f1e2b45550389ecf7c535c7199c6b44b1c6a46303f7b965e508f1',
    '1ef68e93964c2d9230100c1347c328f6385a7bc027879dc3d4c055e6fe80cb3c',
    '259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
    '295682d9eb5acb5c1976a460c085734bfaf38482b0a3f02591c2bfdcd4128549',
    '31f1fbe85b8899258ea5bcf5f93f7ac8238660c386aeab40649c715bd2e38a0a',
    '336d3e4105766f8ad328a7ee9571e743f376f8cbcf6a969ca7e353fe3235c523',
    '38f5cfb55fc8b048e82a5c895b25fefae7a70c71ab9990c535d1030637bf6a1f',
    '3c4c675825f7509877bc10497f498c9a2e3433bf922bd870914a2eb21a54fd26',
    '432f367a4c5b5674de2e2977744d10289a064e5704b21af6607b4975be47c580',
    '43a71aeb641faa18742cb826772a8566c6c947d7050f9ab15459de6cc2b3b6af',
    '44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078',
    '4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2',
    '472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',
    '4be73d68f433869188fe5e7f09c7f681ed51003da6aa5d19ce368726d8e271ee',
    '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac',
    '505bc0a3928d8aef5ce441c5a611fdd32e1e8eccdc15cc3a52b88030acb50f81',
    '519dc0d672d1c295fc69b629af8721ccb1a1f136d1976685a68487e62547ffe0',
    '51c70bb8a299943b27f8b354571272692d8f2705036a1a9562156c76da5f025b',
    '52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde',
    '53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4',
    '550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18',
    '5cee644e5ffbef1ba021c7f389b33bafd3b1841f04d3edd7922d5084c2c4e0c7',
    '648c8ffa496e1716017906d0bf135debfc93386ae86aa3d4adbda9a505985fd9',
    '697a05c6fe4a07c601d46da80885645ad574ea19b47ee795ccff216c9f1f1808',
    '699f2992cd71e2e28cf45f81347ff22e76b37541ce88087742884cd0e9aadc68',
    '78a981bd27ba0c65a9169548665a17bda9f49050d0d3893a6567d1eb92cd003d',
    '7bdb668e6127b7eafc837a883f0648002bd063c736f55a4f673e787250a3fb04',
    '7f4cbe0b36b5d09466476a7d4e01f4f976c67872d549f4ff47b3e1e3a2b403af',
    '8922a6ac8fd0258ec27738ca101867169b20d90a60fc84f93df77acd5bf7c80b',
    '8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab',
    '912a679e4b9b1d1a75170254fd675b8c24b664d80ad7ea7e460241a23535a406',
    '9ab2d381f90b485a68b82bc07f94397a0373e3215ad20935a958738e55f3cfc2',
    '9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab',
    'a4816cc1fb76cb3c5e481186833fc0ae9cf426a1406a2607e974e65e9cddba4f',
    'a984e7fb886aa02e29d112766d3ce26a4f78eac540ce7bbdbd42af2761928f6d',
    'ab298b962a63e4be9582513aaa84a5e270adba5fd2b16a50e59540524f63c3b8',
    'ade080c6618cbbb0a25680cf847f312b5e19b22bfe1cafec0436987ebe5b1e7e',
    'b83d1d77935b6cfd44105b54600ffc4b6bd82de57dec65571bcb117fa8398ba3',
    'bdc789019cee8ddfae20d5f769299993b4b330b2d38d1218646cf89e77fbbd4d',
    'c8e79ff4ac55f4b772057de28e539727b7f4f2a3de73bf7a082a0ace86d609eb',
    'ca20076870e8fb604e61802605a9ac45419c82dd3e23404c56c4869f9502a5ef',
    'd616d323a9eeb9da1b66f8d5df671d63c092c9919cb2c0b223e29c63257c944d',
    'd6eb7ce7723e2f6dc13b90b41a29ded27dbd815bad633fdf582447c686018896',
    'd8d4bf68a76e4e4c5f21de7ac613451f7115a04db686151e78b8ec0b6a22022b',
    'da6c593410340b19bb212b9f6d274f95b08c0fc8f2570cd66bc5ed42c560acab',
    'dab46d798d29aff2e99c23f47ed3064f5cafb1644629b015c95a2dd2ee593bb4',
    'df40099c6306ca1f47fcc8a62e2fa39486d4e223177afdc51b2ad189691802d8',
    'e17b7aedd251a016c01ef9158e6e4aa940d9f1b35942d86028dc1222192a9258',
    'eea70a7948d25a9a791dbcb39228af4ea4049fe5ebdee9c04884be8cca3da835',
    'f0d0ab13ff53adc3c4d57e95a5f83d80b06f2cbc0bf002b52cf7b496612e0ce4',
    'f5effed21f671bbf4551ecebb7fe95f3be1cf09c16a60afe64d2f0b95be9d1eb',
    'fac507fa4d1649e8b24c195d990f1fc3ca3633d917839e1751a9d412a14ab5e3',
    'fe9adb627a6f45747c5a8223b671774791ededf9364f6544be487c540107fa4f',
]


def get_id_and_rle(out_dir):
    npy_dir = os.path.join(out_dir, 'submit', 'npys')
    csv_file = os.path.join(out_dir, 'submit', 'submit.csv')
    return run_npy_to_sumbit_csv(npy_dir=npy_dir, csv_file=csv_file)


# automatic pipeline
def generate_combined_csv():
    cvs_ImageId_purple, cvs_EncodedPixels_purple = get_id_and_rle('./mask_rcnn_ensemble_purple_to_purple')
    cvs_ImageId_gray, cvs_EncodedPixels_gray = get_id_and_rle('./mask_rcnn_ensemble_gray_to_gray_419')
    cvs_ImageId = cvs_ImageId_purple + cvs_ImageId_gray
    cvs_EncodedPixels = cvs_EncodedPixels_purple + cvs_EncodedPixels_gray
    df = pd.DataFrame({'ImageId': cvs_ImageId, 'EncodedPixels': cvs_EncodedPixels})
    df.to_csv('submission_combined.csv', index=False, columns=['ImageId', 'EncodedPixels'])


def predict_and_generate_csv(out_dir, initial_checkpoint, data_dir, image_set, image_folder, color_scheme, test_augment_mode):
    run_submit(
        out_dir=out_dir,
        initial_checkpoint=initial_checkpoint,
        data_dir=data_dir,
        image_set=image_set,
        image_folder=image_folder,
        color_scheme=color_scheme,
        test_augment_mode=test_augment_mode)
    get_id_and_rle(out_dir)


def combine_csvs(combined_file_name, *args):
    ids = []
    rles = []
    for csv in args[0]:
        for id, rle in zip(csv['ImageId'], csv['EncodedPixels']):
            if type(rle) == str:
                ids.append(id)
                rles.append(rle)
            else:
                print('not string')
    # kaggle submission requires all test image to be listed!
    for t in ALL_TEST_IMAGE_ID:
        ids.append(t)
        rles.append('')  # null
    df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
    df.to_csv(combined_file_name, index=False, columns=['ImageId', 'EncodedPixels'])


def main():
    # data_dir = '../data'
    # image_set = 'test1_ids_all_65'
    #
    # prediction, locs = classify(data_dir=data_dir, image_set=image_set)
    #
    # generate_class_file(prediction, locs, ['gray', 'purple'])

    # out_dir = './mask_rcnn_ensemble_gray_to_purple_ly'
    # out_dir = './mask_rcnn_ensemble_gray_to_gray_419'
    # out_dir = './mask_rcnn_ensemble_purple_to_purple'

    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_purple108/00036500_model.pth'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500_ly/00009000_model.pth'

    # image_set = '../ensemble/purple'
    # run_submit(out_dir=out_dir, initial_checkpoint=initial_checkpoint, data_dir=data_dir, image_set=image_set)

    # npy_dir = os.path.join(out_dir, 'submit', 'npys')
    # csv_file = os.path.join(out_dir, 'submit', 'submit.csv')
    # run_npy_to_sumbit_csv(image_dir=None, npy_dir=npy_dir, csv_file=csv_file)

    # data_dir = '../data'
    # image_set = '../ensemble/gray'
    # out_dir = './mask_rcnn_gray_model_on_gray_linear_normalized_enhance'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
    # run_submit(out_dir=out_dir, initial_checkpoint=initial_checkpoint, data_dir=data_dir, image_set=image_set)
    # cvs_ImageId_purple, cvs_EncodedPixels_purple = get_id_and_rle(out_dir)

    # data_dir = '../data'
    # image_set = '../ensemble/purple_yellow'
    # out_dir = './mask_rcnn_gray_model_on_purple_yellow_square_enhance'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
    # run_submit(out_dir=out_dir, initial_checkpoint=initial_checkpoint, data_dir=data_dir, image_set=image_set)
    #
    # cvs_ImageId_purple, cvs_EncodedPixels_purple = get_id_and_rle('./mask_rcnn_gray_model_on_purple_yellow_square_enhance')

    # data_dir = '../data'
    # image_set = '../ensemble/purple_purple'
    # out_dir = './mask_rcnn_purple_model_on_purple_purple'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_purple108/00036500_model.pth'
    # run_submit(out_dir=out_dir, initial_checkpoint=initial_checkpoint, data_dir=data_dir, image_set=image_set)
    # cvs_ImageId_purple, cvs_EncodedPixels_purple = get_id_and_rle('./mask_rcnn_purple_model_on_purple_purple')

    # data_dir = '../data'
    # image_set = '../ensemble/purple'
    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_purple-purple'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-3_purple/00011000_model.pth'
    # predict_and_generate_csv(data_dir, image_set, out_dir, initial_checkpoint)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-4_gray500/00047500_model.pth'
    # data_dir = '../data'
    # image_set = 'gray'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_GRAYSCALE
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_external_H&E'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-6_external_H&E/00010500_model.pth'
    # data_dir = '../data'
    # image_set = 'purple_purple'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_COLOR
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_purple_yellow'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-4_gray500/00011000_model.pth'
    # data_dir = '../data'
    # image_set = 'purple_yellow'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_COLOR
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray-yellow'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-4_gray500/00033500_model.pth'
    # data_dir = '../data'
    # image_set = 'purple_yellow'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_GRAYSCALE
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # combine_csvs(
    #     datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray-gray.csv',
    #     [
    #         pd.read_csv(out_dir + '/submit/submit.csv'),
    #         pd.read_csv('2018-04-03_11-43-01_mask_rcnn_purple-purple/submit/submit.csv')
    #     ])

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_external-yellow'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-6_external_H&E/00010500_model.pth'
    # data_dir = '../data'
    # image_set = 'purple_yellow'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_COLOR
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # combine_csvs(
    #     datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray-gray+purple-yellow+external.csv',
    #     [
    #         pd.read_csv('2018-04-05_16-02-41_mask_rcnn_gray/submit/submit.csv'),
    #         pd.read_csv('2018-04-06_15-26-32_mask_rcnn_gray-yellow/submit/submit.csv'),
    #         pd.read_csv('2018-04-06_15-05-18_mask_rcnn_external_H&E/submit/submit.csv')
    #     ])

    # combine_csvs(
    #     'predictions/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_external_test.csv',
    #     [
    #         pd.read_csv('predictions/2018-04-06_15-05-18_mask_rcnn_external_H&E/submit/submit.csv'),
    #     ])

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray-gray'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-4_gray500/00052500_model.pth'
    # data_dir = '../data'
    # image_set = 'gray'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_GRAYSCALE
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_purple-yellow-transformed'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-3_purple/00011000_model.pth'
    # data_dir = '../data'
    # image_set = 'test_ids_yellow_8'
    # image_folder = 'enhanced_color_pur_yel2'
    # color_scheme = cv2.IMREAD_COLOR
    # predict_and_generate_csv(
    #     out_dir=out_dir,
    #     initial_checkpoint=initial_checkpoint,
    #     data_dir=data_dir,
    #     image_set=image_set,
    #     image_folder=image_folder,
    #     color_scheme=color_scheme)

    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-4_gray500/00033500_model.pth'
    # data_dir = '../data'
    # image_set = 'gray'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_GRAYSCALE
    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_maskrcnn_gray-gray_test_augment/'
    # for test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none']:
    #     predict_and_generate_csv(
    #         out_dir=out_dir + test_augment_mode,
    #         initial_checkpoint=initial_checkpoint,
    #         data_dir=data_dir,
    #         image_set=image_set,
    #         image_folder=image_folder,
    #         color_scheme=color_scheme,
    #         test_augment_mode=test_augment_mode)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_external-purple/'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-6_external_H&E/00015500_model.pth'
    # data_dir = '../data'
    # image_set = 'test_ids_purple_4'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_COLOR
    # for test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none']:
    #     predict_and_generate_csv(
    #         out_dir=out_dir + test_augment_mode,
    #         initial_checkpoint=initial_checkpoint,
    #         data_dir=data_dir,
    #         image_set=image_set,
    #         image_folder=image_folder,
    #         color_scheme=color_scheme,
    #         test_augment_mode=test_augment_mode)

    # out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_purple-yellow/'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-7_purple108/00017000_model.pth'
    # data_dir = '../data'
    # image_set = 'test_ids_yellow_8'
    # image_folder = 'stage1_test'
    # color_scheme = cv2.IMREAD_COLOR
    # for test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none']:
    #     predict_and_generate_csv(
    #         out_dir=out_dir + test_augment_mode,
    #         initial_checkpoint=initial_checkpoint,
    #         data_dir=data_dir,
    #         image_set=image_set,
    #         image_folder=image_folder,
    #         color_scheme=color_scheme,
    #         test_augment_mode=test_augment_mode)

    # combine_csvs(
    #     datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mask_rcnn_gray-gray-augment+purple-yellow+external.csv',
    #     [
    #         pd.read_csv('2018-04-07_15-30-34_maskrcnn_gray-gray_test_augment/submit/submit.csv'),
    #         pd.read_csv('2018-04-07_20-34-25_mask_rcnn_external-purple/submit/submit.csv'),
    #         pd.read_csv('2018-04-08_00-45-40_mask_rcnn_purple-yellow/submit/submit.csv')
    #     ])

    out_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_gray_model/'
    initial_checkpoint = '../gcp_data_sync/latest_model.pth'
    # initial_checkpoint = '../aws_data_sync/2018-4-14_gray800_box/2018-04-14_18-16-22/latest_model.pth'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-13_abn_gray2rgb/abn_gray2rgb_model.pth'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
    # initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-4-7_purple108/00017000_model.pth'
    data_dir = '../data/2018-4-12_dataset'
    # data_dir = '../data'
    # image_set = 'test2_color_key.txt'
    # image_set = 'test1_ids_all_65'
    image_set = 'test2_color_key.txt'
    image_folder = '../stage2_test'
    # image_folder = 'stage1_test'
    color_scheme = cv2.IMREAD_GRAYSCALE
    # color_scheme = cv2.IMREAD_COLOR
    # for test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none']:
    #     predict_and_generate_csv(
    #         out_dir=out_dir + test_augment_mode,
    #         initial_checkpoint=initial_checkpoint,
    #         data_dir=data_dir,
    #         image_set=image_set,
    #         image_folder=image_folder,
    #         color_scheme=color_scheme,
    #         test_augment_mode=test_augment_mode)
    predict_and_generate_csv(
        out_dir=out_dir + 'none',
        initial_checkpoint=initial_checkpoint,
        data_dir=data_dir,
        image_set=image_set,
        image_folder=image_folder,
        color_scheme=color_scheme,
        test_augment_mode='none')


if __name__ == '__main__':
    main()
