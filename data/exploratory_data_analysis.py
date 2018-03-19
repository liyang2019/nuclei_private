import pathlib
import imageio
import numpy as np

import matplotlib.pyplot as plt

# Glob the training data and load a single image path
training_paths = pathlib.Path('stage1_train').glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])
image_to_segmentation = {}
for p in training_sorted:
    image_key = str(p).split('/')[-1].strip('.png')
    seg_paths = pathlib.Path('stage1_train/' + image_key).glob('masks/*.png')
    seg_sorted = sorted([x for x in seg_paths])
    image_to_segmentation[image_key] = []
    for p_seg in seg_sorted:
        image_to_segmentation[image_key].append(str(p_seg).split('/')[-1].strip('.png'))


# im = imageio.imread(str(im_path))