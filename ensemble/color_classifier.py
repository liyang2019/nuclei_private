import os
import imageio
import numpy as np
from sklearn.cluster import KMeans


def get_train_path(data_folder, image_key):
    return os.path.join(data_folder, 'stage1_images/stage1_train', image_key + '.png')


def get_test_path(data_folder, image_key):
    return os.path.join(data_folder, 'stage1_images/stage1_test', image_key + '.png')


def get_average_RGB(data_folder, image_keys, mode):
    num_images = len(image_keys)
    RGB_intensities = np.zeros((num_images, 3))
    for i in range(num_images):
        if mode in ['train']:
            image_path = get_train_path(data_folder, image_keys[i])
        elif mode in ['test']:
            image_path = get_test_path(data_folder, image_keys[i])
        else:
            raise NotImplementedError
        im = imageio.imread(image_path)[:, :, :3]
        RGB_intensities[i, :] = im.reshape(-1, 3).mean(0)
    return RGB_intensities


def classify(data_folder, image_set):
    # Glob the training data and load a single image path
    image_test_set = os.path.join(data_folder, 'image_sets', image_set)
    image_test_locations = [x.strip('\n') for x in open(image_test_set, 'r').readlines() if x is not '\n']
    image_test_keys = [x.split('/')[-1].strip('.png') for x in image_test_locations]
    print('the total number of training image is: ', len(image_test_keys))
    RGB_test_intensities = get_average_RGB(data_folder, image_test_keys, 'test')
    model = KMeans(n_clusters=2, max_iter=1000, tol=1e-6)
    prediction = model.fit_predict(RGB_test_intensities)
    # put 0 for black and white image, and 1 for color image
    mask = prediction > 0
    if RGB_test_intensities[mask, :].mean() < RGB_test_intensities[~mask, :].mean():
        prediction ^= 1
    return prediction, image_test_locations


def generate_class_file(preds, locs, files):
    """
    :param preds: The predictions, 0 for black and white image, 1 for purple image
    :param locs: The image locations for two classes.
    :param files: The files to save the image locations.
    """
    with open(files[0], 'w') as c1, open(files[1], 'w') as c2:
        for i in range(len(preds)):
            if preds[i] == 0:
                print(locs[i], file=c1)
            else:
                print(locs[i], file=c2)

if __name__ == '__main__':
    pass