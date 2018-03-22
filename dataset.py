import os
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import torch
from skimage import transform


class SemanticSegmentationDataset(Dataset):

    def __init__(self, root, imgkey_file, img_folder, crop_size, validation=True, testing=False):
        """
        Initialization.
        :param root: The folder root of the image samples.
        :param imgkey_file: The filename of the image keys txt file.
        :param img_folder: The folder name containing images and flattened segmentation.
        :param crop_size: The image and segmentation size after scale and crop for training.
        :param validation: True if is training.
        """
        self.root = root
        self.size = crop_size
        self.validation = validation
        self.testing = testing

        # mean and std using ImageNet mean and std,
        # as required by http://pytorch.org/docs/master/torchvision/models.html
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        imgkey_file = os.path.join(root, imgkey_file)
        img_folder = os.path.join(root, img_folder)
        self.img_keys = [x.strip('\n') for x in open(imgkey_file, 'r').readlines() if x is not '\n']
        self.img_paths = [os.path.join(img_folder, k + '.png') for k in self.img_keys]
        self.seg_paths = [os.path.join(img_folder, k + '_seg.png') for k in self.img_keys]

    def _scale_and_crop(self, img, seg, crop_size):
        """
        scale crop the image to make every image of the same square size, H = W = crop_size
        :param img: The image.
        :param seg: The segmentation of the image.
        :param crop_size: The crop size.
        :return: The cropped image and segmentation.
        """
        h, w = img.shape[:2]
        scale = crop_size / min(h, w)
        if scale > 1:
            print('scale: ', scale)
            img = transform.rescale(img, scale, mode='reflect', order=1)  # order 1 is bilinear
            if not self.testing:
                seg = transform.rescale(seg.astype(np.float), scale, mode='reflect', order=0)  # order 0 is nearest neighbor

        h_s, w_s = img.shape[0], img.shape[1]
        if self.validation or self.testing:
            # center crop
            x1 = (w_s - crop_size) // 2
            y1 = (h_s - crop_size) // 2
        else:
            # random crop
            x1 = random.randint(0, w_s - crop_size)
            y1 = random.randint(0, h_s - crop_size)

        img_crop = img[y1: y1 + crop_size, x1: x1 + crop_size, :]
        if not self.testing:
            seg_crop = seg[y1: y1 + crop_size, x1: x1 + crop_size]
            return img_crop, seg_crop
        else:
            return img_crop, None

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
        img_key = self.img_keys[index]
        img_path = self.img_paths[index]
        img = imageio.imread(img_path)[:, :, :3]  # remove alpha channel
        seg = None
        if not self.testing:
            seg_path = self.seg_paths[index]
            seg = imageio.imread(seg_path) > 0
            if self.size > 0:
                img, seg = self._scale_and_crop(img, seg, self.size)
                if random.choice([-1, 1]) > 0:
                    img, seg = self._flip(img, seg)
            seg = seg.astype(np.long)
            seg = torch.from_numpy(seg.copy())
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).contiguous()
        return img, seg, img_key

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.img_keys)


def main():
    dataset = SemanticSegmentationDataset(
        root='kaggle_train_data',
        imgkey_file='image_train.txt',
        img_folder='stage1_train_imgs_and_flattenedmasks',
        crop_size=224, validation=True, testing=False)

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
