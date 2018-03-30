import pandas as pd

from color_classifier import *
from mask_rcnn.submit import run_submit, run_npy_to_sumbit_csv

data_folder = '../data'
image_set = 'test1_ids_all_65'

prediction, locs = classify(data_folder=data_folder, image_set=image_set)

generate_class_file(prediction, locs, ['blackwhite', 'purple'])

# out_dir = './mask_rcnn_ensemble_gray_to_purple_ly'
# out_dir = './mask_rcnn_ensemble_gray_to_gray_419'
out_dir = './mask_rcnn_ensemble_purple_to_purple'

# initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_purple108/00036500_model.pth'
# initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
# initial_checkpoint = '/Users/li/saved_model_macbook_local/2018-3-30_maskrcnn_gray500_ly/00009000_model.pth'

# image_set = '../ensemble/purple'
# run_submit(out_dir=out_dir, initial_checkpoint=initial_checkpoint, data_dir=data_folder, image_set=image_set)

# npy_dir = os.path.join(out_dir, 'submit', 'npys')
# csv_file = os.path.join(out_dir, 'submit', 'submit.csv')
# run_npy_to_sumbit_csv(image_dir=None, npy_dir=npy_dir, csv_file=csv_file)


def get_id_and_rle(out_dir):
    npy_dir = os.path.join(out_dir, 'submit', 'npys')
    csv_file = os.path.join(out_dir, 'submit', 'submit.csv')
    return run_npy_to_sumbit_csv(image_dir=None, npy_dir=npy_dir, csv_file=csv_file)


# automatic pipeline
def generate_combined_csv():
    cvs_ImageId_purple, cvs_EncodedPixels_purple = get_id_and_rle('./mask_rcnn_ensemble_purple_to_purple')
    cvs_ImageId_gray, cvs_EncodedPixels_gray = get_id_and_rle('./mask_rcnn_ensemble_gray_to_gray_419')
    cvs_ImageId = cvs_ImageId_purple + cvs_ImageId_gray
    cvs_EncodedPixels = cvs_EncodedPixels_purple + cvs_EncodedPixels_gray
    df = pd.DataFrame({'ImageId': cvs_ImageId, 'EncodedPixels': cvs_EncodedPixels})
    df.to_csv('submission_combined.csv', index=False, columns=['ImageId', 'EncodedPixels'])


if __name__ == '__main__':
    generate_combined_csv()
