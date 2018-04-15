import argparse

from ensemble_submit import *


def ensemble_predict(initial_checkpoint, identifier, test_augment_mode):
    model = initial_checkpoint.split('/')[-1].strip('.pth')
    out_dir = identifier + '_ensemble_' + model
    print('augment in mode: ' + test_augment_mode)
    predict_and_generate_csv(
        out_dir=os.path.join(out_dir, test_augment_mode),
        initial_checkpoint=initial_checkpoint,
        data_dir='../data/2018-4-12_dataset',
        image_set='test2_gray_key.txt',
        # image_set='gray_debug.txt',
        image_folder='../stage2_test',
        color_scheme=cv2.IMREAD_GRAYSCALE,
        test_augment_mode=test_augment_mode)


def main():
    parser = argparse.ArgumentParser(description='Script to run segmentation models prediction')
    parser.add_argument('--initial_checkpoint', action='store', default=' ')
    parser.add_argument('--test_augment_mode', action='store', default='none')

    args = parser.parse_args()

    initial_checkpoint = args.initial_checkpoint
    identifier = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    test_augment_mode = args.test_augment_mode
    ensemble_predict(initial_checkpoint, identifier, test_augment_mode)


if __name__ == '__main__':
    main()
