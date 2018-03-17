import os
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import transform


class kagglebowl18_dataset(Dataset):

    def __init__(self, root, file, segfile, classfile, size, max_sample=-1, train=True):
        """
        Initialization.
        :param root: The folder root of the image samples.
        :param file: The filename of the image samples txt file.
        :param segfile: The filename of the image segmentation txt file.
        :param classfile: The file that has class index information.
        :param size: The image and segmentation size after scale and crop for training.
        :param max_sample: The max number of samples.
        :param train: True if is training.
        """
        self.root = root
        self.size = size
        self.train = train

        # mean and std using ImageNet mean and std,
        # as required by http://pytorch.org/docs/master/torchvision/models.html
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        file = os.path.join(root, file)
        segfile = os.path.join(root, segfile)
        classfile = os.path.join(root, classfile)
        self.list_sample = [x.strip('\n') for x in open(file, 'r').readlines() if x is not '\n']
        self.list_sample_segm = [x.strip('\n') for x in open(segfile, 'r').readlines() if x is not '\n']
        self.class_dict = {0: 0}
        idx_new = 1
        for idx in open(classfile, 'r').readlines():
            if idx is not '\n':
                self.class_dict[int(idx)] = idx_new
                idx_new += 1

        # if self.train:
        #  random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0: max_sample]
            self.list_sample_segm = self.list_sample_segm[0: max_sample]
        num_sample = len(self.list_sample)

        assert num_sample > 0

    @staticmethod
    def _scale_and_crop(img, seg, crop_size, train):
        """
        scale crop the image to make every image of the same square size, H = W = crop_size
        :param img: The image.
        :param seg: The segmentation of the image.
        :param crop_size: The crop size.
        :param train: True if is training.
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
            seg = transform.rescale(seg, scale, mode='reflect', order=0)  # order 0 is nearest neighbor

        h_s, w_s = img.shape[0], seg.shape[1]
        if train:
            # random crop
            x1 = random.randint(0, w_s - crop_size)
            y1 = random.randint(0, h_s - crop_size)
        else:
            # center crop
            x1 = (w_s - crop_size) // 2
            y1 = (h_s - crop_size) // 2

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
        # index %= 1  # one image can generate 10 crops.
        name_img = self.list_sample[index]
        name_seg = self.list_sample_segm[index]
        path_img = os.path.join(self.root, name_img)
        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
        # print(path_seg)
        for path_segm_temp in name_seg.split(","):
            path_segm_temp = os.path.join(self.root, path_segm_temp)
            if path_segm_temp is not "":
                # print(path_segm_temp)
                assert os.path.exists(path_segm_temp), '[{}] does not exist'.format(path_segm_temp)
        # load image and label
        img = imageio.imread(path_img)[:, :, :3]
        seg = np.full((img.shape[0], img.shape[1]), 0)
        for path_segm_temp in name_seg.strip('\n').split(","):
            path_segm_temp = os.path.join(self.root, path_segm_temp)
            seg_t = imageio.imread(path_segm_temp)
            seg |= seg_t > 0

        # plt.imshow(seg)
        # plt.show()
        # plt.imshow(img)
        # plt.show()
        assert (img.ndim == 3)
        assert (seg.ndim == 2)
        assert (img.shape[0] == seg.shape[0])
        assert (img.shape[1] == seg.shape[1])
        # print(img.ndim)
        # random scale, crop, flip
        if self.size > 0:
            img, seg = self._scale_and_crop(img, seg, self.size, self.train)
            if random.choice([-1, 1]) > 0:
                img, seg = self._flip(img, seg)

        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        # seg[i, j] = 0 for unlabeled pixels.
        # print(seg[:,:]>0)
        # seg[seg[:, :] > 0] = 1

        # to torch tensor

        image = torch.from_numpy(img.copy())
        segmentation = torch.from_numpy(seg.copy())
        image = self.img_transform(image)

        return image, segmentation, path_img

    def __len__(self):
        """
    Get the length of the dataset.
    :return: The length of the dataset.
    """
        return len(self.list_sample)
        # return 10


def main():
    dataset = kagglebowl18_dataset(
        root='kaggle_train_data',
        file='image_train.txt',
        segfile='segmentation_train.txt',
        classfile='class_train.txt',
        size=224, max_sample=-1, train=True)

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
