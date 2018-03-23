import time
import torch
from torch.autograd import Variable
import numpy as np

torch.set_num_threads(16)


class Trainer:
    def __init__(self, cuda, model, train_loader, val_loader, loss, optimizer,
                 n_epochs, n_save, n_print, learning_rate, is_validation, lr_decay_every, lr_decay_ratio):
        """
        Initialization for Trainer of FCN model for image segmentation.
        :param cuda: True is cuda available.
        :param model: The end-to-end FCN model.
        :param train_loader: The training dataset loader.
        :param val_loader: The validation dataset loader.
        :param loss: The loss function
        :param optimizer: The optimizer.
        :param n_epochs: The max number for epochs.
        :param n_save; Save every n_save iterations.
        :param n_print: Print every n_print iterations.
        :param learning_rate: The learning rate.
        :param is_validation: If using validation.
        :param lr_decay_every: learning rate decay every lr_decay_every steps.
        :param lr_decay_ratio: learning rate decay ratio.
        """
        self.cuda = cuda
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = loss
        self.n_epochs = n_epochs
        self.n_save = n_save
        self.n_print = n_print
        self.optimizer = optimizer
        self.n_iter = 0
        self.is_validation = is_validation
        self.learning_rate = learning_rate
        self.lr_decay_every = lr_decay_every
        self.lr_decay_ratio = lr_decay_ratio

    def train(self):
        n_iter = 0
        loss_train_print = 0
        train_count = 0
        loss_val_print = 0
        val_count = 0
        for epoch in range(self.n_epochs):
            tic = time.time()
            for img, seg, _ in self.train_loader:
                n_iter += 1
                self.optimizer.zero_grad()
                if self.cuda:
                    img, seg = img.cuda(), seg.cuda()
                img, seg = Variable(img), Variable(seg)
                output = self.model(img)

                # forward
                loss = self.loss(output, seg)
                loss_train_print += float(loss.data)
                train_count += 1

                # backward
                loss.backward()

                # update parameter
                self.optimizer.step()

                # validation
                if self.is_validation:
                    loss_val = self.validation_loss()
                    loss_val_print += loss_val
                    val_count += 1

                # learning rate decay
                if ((n_iter + 1) % self.lr_decay_every) == 0:
                    self.learning_rate *= self.lr_decay_ratio

                # print to log
                if ((n_iter + 1) % self.n_print) == 0:
                    toc = time.time()
                    print("epoch {} | step {} | loss_train {} | loss_val {} | lr {} | time {} "
                          .format(epoch, self.n_iter, loss_train_print / train_count, loss_val_print / (val_count + 1e-16), self.learning_rate, toc - tic))
                    with open('log.txt', 'a') as log:
                        print("epoch {} | step {} | loss_train {} | loss_val {} | lr {} | time {} "
                              .format(epoch, self.n_iter, loss_train_print / train_count, loss_val_print / (val_count + 1e-16), self.learning_rate, toc - tic), file=log)
                    tic = time.time()
                    loss_train_print = 0
                    train_count = 0
                    loss_val_print = 0
                    val_count = 0

                # save model
                if ((n_iter + 1) % self.n_save) == 0:
                    torch.save(self.model, 'model_saved.pt')

    def validation_loss(self):
        """
        Calculate validation loss
        :return: the average loss over validation set.
        """
        loss_val = 0
        count = 0
        for img, seg, _ in self.val_loader:
            img, seg = Variable(img), Variable(seg)
            if self.cuda:
                img = img.cuda()
                seg = seg.cuda()
            output_val = self.model(img)
            loss_val += self.loss(output_val, seg).data
            count += 1
        loss_val /= count
        return loss_val


def main():
    pass


if __name__ == '__main__':
    main()
