from torch import optim
import torch
from dataset import mykaggle_loader
from torch.utils.data import DataLoader
from fcn8s import FCN8s
from fcn32s import FCN32s
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
class Trainer:
  def __init__(self, cuda, model, train_loader, val_loader, loss, n_epochs, n_save, learning_rate,is_validation):
    """
    Initialization for Trainer of FCN model for image segmentation.
    :param cuda: True is cuda available.
    :param model: The end-to-end FCN model.
    :param train_loader: The training dataset loader.
    :param val_loader: The validation dataset loader.
    :param loss: The loss function
    :param n_epochs: The max number for epochs.
    :param n_save; Save every n_save iterations.
    :param learning_rate: The learning rate.
    :param is_validation: If using validation
    """
    self.cuda = cuda
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.loss = loss
    self.n_epochs = n_epochs
    self.n_save = n_save
    self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    self.n_iter = 0
    self.is_validation = is_validation

  def train_epoch(self):
    for img, seg, _ in self.train_loader:
      #print(img.numpy().shape)
      #print(seg.numpy().shape)
      self.n_iter += 1
      self.optimizer.zero_grad()
      if self.cuda:
        img, seg = img.cuda(), seg.cuda()
      img, seg = Variable(img), Variable(seg)
      output = self.model(img)

      # forward
      #print(seg)
      loss = self.loss(output, seg)
      #loss /= len(img)

      #backward
      loss.backward()
      self.optimizer.step()

      print(self.n_iter, float(loss.data))

    # validation
    if (self.is_validation):
      for img,seg,_ in self.val_loader:
        img, seg = Variable(img), Variable(seg)
        output_val = self.model(img)
        loss_val = self.loss(output_val,seg)
        print(self.n_iter,float(loss_val.data))


  def train(self):
    for epoch in range(self.n_epochs):
      print('epoch', epoch)
      self.train_epoch()

  # @staticmethod
  # def loss(input, target, size_average=True):
  #   """
  #   The cross entropy loss function.
  #   :param size_average: True if loss averaged over minibatch.
  #   :param input: The output of the model, (n, c, h, w)
  #   :param target: The target, (n, h, w)
  #   :return: The loss.
  #   """
  #   n, c, h, w = input.size()
  #   log_p = F.log_softmax(input, dim=1)
  #   # log_p: (n*h*w, c)
  #   log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
  #   log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) > 0]
  #   log_p = log_p.view(-1, c)
  #   # target: (n*h*w,)
  #   mask = target > 0
  #   target = target[mask]
  #   loss = F.nll_loss(log_p, target, size_average=False)
  #   if size_average:
  #     loss /= mask.data.sum()
  #   return loss


def main():
  cuda = torch.cuda.is_available()

  num_classes = 2
  pretrained = True
  image_size = 128
  batch_size = 1
  n_epochs = 30
  n_save = 10
  learning_rate = 1e-3
  is_validation = False

  fcn = FCN8s(num_classes=num_classes, pretrained=pretrained)
  fcn = fcn.cuda() if cuda else fcn
  loss = torch.nn.CrossEntropyLoss()
  train_set = mykaggle_loader('kaggle_train_data', 'kaggle_train_data/train_kaggle1.txt',
                              'kaggle_train_data/train_kaggle1_segm.txt', 'kaggle_train_data/train_kaggle1_class.txt',
                              128)
  val_set = mykaggle_loader('kaggle_train_data', 'kaggle_train_data/train_kaggle2.txt',
                              'kaggle_train_data/train_kaggle2_segm.txt', 'kaggle_train_data/train_kaggle2_class.txt',
                              128)

  train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
  val_loader = DataLoader(val_set, len(val_set))

  trainer = Trainer(cuda, fcn, train_loader, val_loader,loss, n_epochs, n_save, learning_rate,is_validation)
  trainer.train()
  for img,seg,_ in train_loader:
    res = trainer.model(Variable(img))
    #print(res)
    res = res.data.numpy()
    #print(res.shape)
    res = np.squeeze(res[0,:,:,:])
    res1 = np.squeeze(res[0,:,:])
    res2 = np.squeeze(res[1,:,:])
    #print(res1.shape)
    #print(res2.shape)
    plt.imshow(res1)
    plt.show()
    plt.imshow(res2)
    plt.show()

    seg = seg.numpy()
    seg = seg[0,:,:]
    #print(seg)
    plt.imshow(seg)
    plt.show()


if __name__ == '__main__':
  main()
