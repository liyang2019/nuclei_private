import time
import torch
from torch.autograd import Variable
torch.set_num_threads(16)


class Trainer:
    def __init__(self, cuda, model, train_loader, val_loader, loss, optimizer,
                 n_epochs, n_save, n_print, learning_rate, is_validation):
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
        :param is_validation: If using validation
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

    def train_epoch(self):
        tic = time.time()
        for img, seg, _ in self.train_loader:
            self.n_iter += 1
            self.optimizer.zero_grad()
            if self.cuda:
                img, seg = img.cuda(), seg.cuda()
            img, seg = Variable(img), Variable(seg)
            output = self.model(img)

            # forward
            loss = self.loss(output, seg)

            # backward
            loss.backward()

            # update parameter
            self.optimizer.step()

            # validation
            loss_val = None
            if self.is_validation:
                loss_val = self.validation_loss()

            # print to log
            if ((self.n_iter + 1) % self.n_print) == 0:
                toc = time.time()
                print("step {} | loss_train {} | loss_val {} | lr {} | time {} "
                      .format(self.n_iter, float(loss.data), loss_val, self.learning_rate, toc - tic))
                with open('log.txt', 'a') as log:
                    print("step {} | loss_train {} | loss_val {} | lr {} | time {} "
                          .format(self.n_iter, float(loss.data), loss_val, self.learning_rate, toc - tic), file=log)
                tic = time.time()

            # save model
            if ((self.n_iter + 1) % self.n_save) == 0:
                torch.save(self.model, 'model_saved.pt')

    def train(self):
        for epoch in range(self.n_epochs):
            # print('epoch', epoch)
            self.train_epoch()

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
