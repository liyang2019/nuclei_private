import os

# generate training data
data_dir = "stage1_train"
count = 670
with open("image_train.txt", "w") as image_train, \
        open("class_train.txt", "w") as class_train, \
        open("segmentation_train.txt", "w") as segmentation_train:
    for dir in sorted(os.listdir(data_dir))[0: count + 1]:
        if dir == '.DS_Store':
            continue
        image_dirs = os.path.join(data_dir, dir, "images")
        image_location = os.listdir(image_dirs)
        image_train.write(os.path.join(image_dirs, image_location[0]) + "\n")
        class_train.write("1\n")
        segmentation_dirs = os.listdir(os.path.join(data_dir, dir, "masks"))
        for i, segmentation_dir in enumerate(segmentation_dirs):
            segmentation_train.write(os.path.join(data_dir, dir, "masks", segmentation_dir))
            if i < len(segmentation_dirs) - 1:
                segmentation_train.write(',')
        segmentation_train.write("\n")

# generate testing data
data_dir = "stage1_test"
count = 65
with open("image_test.txt", "w") as image_test, \
        open("class_test.txt", "w") as class_test:
    for dir in sorted(os.listdir(data_dir))[0: count + 1]:
        if dir == '.DS_Store':
            continue
        image_dirs = os.path.join(data_dir, dir, "images")
        image_location = os.listdir(image_dirs)
        image_test.write(os.path.join(image_dirs, image_location[0]) + "\n")
        class_test.write("1\n")
