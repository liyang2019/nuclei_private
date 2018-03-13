import os
#import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from scipy.misc import imread, imresize
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as myimg

class ADE20KDataSet(Dataset):

  def __init__(self, root, file, segfile,classfile, size, max_sample=-1, train=True):
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

    self.list_sample = [x.strip('\n') for x in open(file, 'r').readlines() if x is not '\n']
    self.list_sample_segm = [x.strip('\n') for x in open(segfile,'r').readlines() if x is not '\n']
    self.class_dict = {0: 0}
    idx_new = 0
    for idx in open(classfile, 'r').readlines():
      if idx is not '\n':
        self.class_dict[int(idx)] = idx_new
        idx_new += 1

    #if self.train:
    #  random.shuffle(self.list_sample)

    if max_sample > 0:
      self.list_sample = self.list_sample[0: max_sample]
      self.list_sample_segm = self.list_sample_segm[0:max_sample]
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

    if train:
      # random scale
      scale = random.random() + 0.5  # 0.5-1.5
      scale = max(scale, 1. * crop_size / (min(h, w) - 1))  # ??
    else:
      # scale to crop size
      scale = 1. * crop_size / (min(h, w) - 1)

    img_scale = imresize(img, scale, interp='bilinear')
    seg_scale = imresize(seg, scale, interp='nearest')

    h_s, w_s = img_scale.shape[0], img_scale.shape[1]
    if train:
      # random crop
      x1 = random.randint(0, w_s - crop_size)
      y1 = random.randint(0, h_s - crop_size)
    else:
      # center crop
      x1 = (w_s - crop_size) // 2
      y1 = (h_s - crop_size) // 2

    img_crop = img_scale[y1: y1 + crop_size, x1: x1 + crop_size, :]
    seg_crop = seg_scale[y1: y1 + crop_size, x1: x1 + crop_size]
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
    img_basename = self.list_sample[index]
    img_basename_segm = self.list_sample_segm[index]
    path_img = os.path.join(self.root, img_basename)
    path_seg = os.path.join(self.root, img_basename_segm)
    assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
    #print(path_seg)
    for path_segm_temp in path_seg.split(","):
      if (path_segm_temp is not ""):
        #print(path_segm_temp)
        assert os.path.exists(path_segm_temp), '[{}] does not exist'.format(path_segm_temp)
    # load image and label
    try:
      img = imread(path_img, mode='RGB')
      seg = 0
      for path_segm_temp in path_seg.split(","):
        if (path_segm_temp is not ""):
          seg_t = imread(path_segm_temp, mode='RGB')
          seg = seg + seg_t
      #plt.imshow(seg)
      #plt.show()
      #plt.imshow(img)
      #plt.show()
      assert (img.ndim == 3)
      assert (seg.ndim == 3)
      assert (img.shape[0] == seg.shape[0])
      assert (img.shape[1] == seg.shape[1])
      #print(img.ndim)
      # random scale, crop, flip
      if self.size > 0:
        img, seg = self._scale_and_crop(img, seg, self.size, self.train)
        if random.choice([-1, 1]) > 0:
          img, seg = self._flip(img, seg)
      #plt.imshow(img)
      #plt.show()
      #plt.imshow(seg)
      #plt.show()
      # image to float
      img = img.astype(np.float32) / 255.
      img = img.transpose((2, 0, 1))

      # segmentation to integer encoding according to
      # the loadAde20K.m file in http://groups.csail.mit.edu/vision/datasets/ADE20K/
      seg = np.round(seg[:, :, 0] / 10.) * 256 + seg[:, :, 1]
      # seg[i, j] = 0 for unlabeled pixels.
      seg = seg.astype(np.int)

      #print("")
      #for i in range(seg.shape[0]):
      #  for j in range(seg.shape[1]):
      #    seg[i, j] = self.class_dict[seg[i, j]]

      # to torch tensor
      #print(img)
      #print(seg)
      image = torch.from_numpy(img)
      segmentation = torch.from_numpy(seg)
      #print(image)
      #print(segmentation)

    except Exception as e:
      print("working")
      print('Failed loading image/segmentation [{}]: {}'.format(path_img, e))
      # dummy data
      image = torch.zeros(3, self.size, self.size)
      segmentation = -1 * torch.ones(self.size, self.size).long()
      return image, segmentation, img_basename

      # substracted by mean and divided by std
    image = self.img_transform(image)

    return image, segmentation, img_basename

  def __len__(self):
    """
    Get the length of the dataset.
    :return: The length of the dataset.
    """
    return len(self.list_sample)


def main():
  ade20k = ADE20KDataSet('kaggle_train_data', 'kaggle_train_data/train_kaggle1.txt','kaggle_train_data/train_kaggle1_segm.txt','kaggle_train_data/train_kaggle1_class.txt', 128)

  image, segmentation, img_basename = ade20k.__getitem__(1)
  #print(image)
  #print(segmentation)

if __name__ == '__main__':
  main()
