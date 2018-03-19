import os
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import torch
from skimage import transform


class kagglebowl18_dataset(Dataset):

    def __init__(self, root, file, segfile, classfile, size, validation=True, testing=False):
        """
        Initialization.
        :param root: The folder root of the image samples.
        :param file: The filename of the image samples txt file.
        :param segfile: The filename of the image segmentation txt file.
        :param classfile: The file that has class index information.
        :param size: The image and segmentation size after scale and crop for training.
        :param validation: True if is training.
        """
        self.root = root
        self.size = size
        self.validation = validation
        self.testing = testing

        # mean and std using ImageNet mean and std,
        # as required by http://pytorch.org/docs/master/torchvision/models.html
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        file = os.path.join(root, file)
        self.list_sample = [x.strip('\n') for x in open(file, 'r').readlines() if x is not '\n']

        if not self.testing:
            segfile = os.path.join(root, segfile)
            classfile = os.path.join(root, classfile)
            self.list_sample_segm = [x.strip('\n') for x in open(segfile, 'r').readlines() if x is not '\n']
            self.class_dict = {0: 0}
            idx_new = 1
            for idx in open(classfile, 'r').readlines():
                if idx is not '\n':
                    self.class_dict[int(idx)] = idx_new
                    idx_new += 1

    def _scale_and_crop(self, img, seg, crop_size):
        """
        scale crop the image to make every image of the same square size, H = W = crop_size
        :param img: The image.
        :param seg: The segmentation of the image.
        :param crop_size: The crop size.
        :return: The cropped image and segmentation.
        """
        h, w = img.shape[0], img.shape[1]
        # if train:
        #     # random scale
        #     scale = random.random() + 0.5  # 0.5-1.5
        #     scale = max(scale, 1. * crop_size / (min(h, w) - 1))  # ??
        # else:
        #     # scale to crop size
        #     scale = 1. * crop_size / (min(h, w) - 1)
        scale = crop_size / min(h, w)
        if scale > 1:
            print('scale: ', scale)
            img = transform.rescale(img, scale, mode='reflect', order=1)  # order 1 is bilinear
            seg = transform.rescale(seg.astype(np.float), scale, mode='reflect', order=0)  # order 0 is nearest neighbor

        h_s, w_s = img.shape[0], seg.shape[1]
        if self.validation or self.testing:
            # center crop
            x1 = (w_s - crop_size) // 2
            y1 = (h_s - crop_size) // 2
        else:
            # random crop
            x1 = random.randint(0, w_s - crop_size)
            y1 = random.randint(0, h_s - crop_size)

        img_crop = img[y1: y1 + crop_size, x1: x1 + crop_size, :]
        seg_crop = seg[y1: y1 + crop_size, x1: x1 + crop_size]
        return img_crop, seg_crop

    @staticmethod
    def _flip(img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip

    def __getitem__(self, index):
        """
        Get image from file.
        :param index: The index of the image.
        :return: The image, segmentation, and the image base name.
        """
        img_dir = self.list_sample[index]
        img_key = img_dir.split('/')[-1]
        print('img: ', img_dir)
        path_img = os.path.join(self.root, img_dir)
        img = imageio.imread(path_img)[:, :, :3]  # remove alpha channel

        seg = np.full((img.shape[0], img.shape[1]), -1)
        if not self.testing:
            seg_dir = self.list_sample_segm[index]
            # load segmentation
            seg = np.full((img.shape[0], img.shape[1]), 0)
            for path_segm_temp in seg_dir.strip('\n').split(","):
                path_segm_temp = os.path.join(self.root, path_segm_temp)
                seg_t = imageio.imread(path_segm_temp)
                seg |= seg_t > 0

        # random scale, crop, flip
        if self.size > 0 and not self.testing:
            img, seg = self._scale_and_crop(img, seg, self.size)
            if random.choice([-1, 1]) > 0:
                img, seg = self._flip(img, seg)

        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        # to torch tensor
        image = torch.from_numpy(img.copy()).contiguous()
        segmentation = torch.from_numpy(seg.copy())
        # image = self.img_transform(image)

        # if testing, segmentation are matrix with all elements -1.
        return image, segmentation, img_key

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.list_sample)


def main():
    dataset = kagglebowl18_dataset(
        root='kaggle_train_data',
        file='image_train.txt',
        segfile='segmentation_train.txt',
        classfile='class_train.txt',
        size=224, validation=True)

    # # for img, seg, _ in dataset:
    # img, seg, _ = dataset[0]
    # img = np.transpose(img, (1, 2, 0))
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(seg)
    # plt.colorbar()
    # plt.show()
    print(dataset.__len__())


if __name__ == '__main__':
    main()
