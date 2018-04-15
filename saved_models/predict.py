from ensemble_submit import *


def ensemble_predict(initial_checkpoint, identifier):
    model = initial_checkpoint.split('/')[-1].strip('.pth')
    out_dir = identifier + '_ensemble_' + model
    for test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none', 'blur']:
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
    initial_checkpoint = 'models_gray690/00027500_model.pth'
    identifier = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ensemble_predict(initial_checkpoint, identifier)


if __name__ == '__main__':
    main()
