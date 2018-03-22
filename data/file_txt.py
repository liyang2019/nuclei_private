import pathlib
import numpy as np

training_paths = pathlib.Path('stage1_train').glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])
image_to_segmentation = {}
training_keys = []
for p in training_sorted:
    image_key = str(p).split('/')[-1].strip('.png')
    training_keys.append(image_key)
    seg_paths = pathlib.Path('stage1_train/' + image_key).glob('masks/*.png')
    seg_sorted = sorted([x for x in seg_paths])
    image_to_segmentation[image_key] = []
    for p_seg in seg_sorted:
        image_to_segmentation[image_key].append(str(p_seg).split('/')[-1].strip('.png'))

print('the total number of training image is: ', len(image_to_segmentation))
num = len(image_to_segmentation)
num_train = int(num * 0.8)
num_val = num - num_train
np.random.shuffle(training_keys)
train_keys = training_keys[:num_train]
val_keys = training_keys[num_train:]


def generate_key_file(imgfile, segfile, classfile, keys, img_to_seg_map):
    with open(imgfile, "w") as imgf, \
            open(segfile, "w") as clsf, \
            open(classfile, "w") as segf:
        for imkey in keys:
            imgf.write(imkey + '\n')
            clsf.write("1\n")
            for i, seg_key in enumerate(img_to_seg_map[imkey]):
                segf.write(seg_key)
                if i < len(img_to_seg_map[imkey]) - 1:
                    segf.write(',')
            segf.write('\n')


# generate training data
generate_key_file('image_train.txt', 'class_train.txt', 'segmentation_train.txt', train_keys, image_to_segmentation)

# generate validation data
generate_key_file('image_val.txt', 'class_val.txt', 'segmentation_val.txt', val_keys, image_to_segmentation)

# generate testing data
testing_paths = pathlib.Path('stage1_test').glob('*/images/*.png')
testing_sorted = sorted([x for x in testing_paths])
testing_keys = []
for p in testing_sorted:
    image_key = str(p).split('/')[-1].strip('.png')
    testing_keys.append(image_key)

with open('image_test.txt', "w") as img_file:
    for img_key in testing_keys:
        img_file.write(img_key + '\n')
