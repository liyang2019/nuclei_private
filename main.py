import argparse
import random
import numpy as np

import imageio
import os
import torch
from torch.utils.data import DataLoader
from torch import optim

from dataset import kagglebowl18_dataset
from model.fcn16s import FCN16VGG
from model.fcn32s import FCN32s
from model.fcn8s import FCN8s
from model.unet import UNet
from trainer import Trainer
from submitor import Submitor

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run segmentation models')
    parser.add_argument('--debug', help='Debug the model', action='store_true', default=True)
    parser.add_argument('--use_gpu', help='Debug the model', action='store_true', default=False)
    parser.add_argument('--batch_size', help='desired batch size for training', action='store', type=int,
                        dest='batch_size', default=1)
    parser.add_argument('--num_classes', help='number of classes for prediction', action='store', type=int,
                        dest='num_classes', default=2)
    parser.add_argument('--output_dir', help='path to saving outputs', action='store', dest='output_dir',
                        default='output')
    parser.add_argument('--model', help='model to train on', action='store', dest='model', default='unet')
    parser.add_argument('--learning_rate', help='starting learning rate', action='store', type=float,
                        dest='learning_rate', default=0.001)
    parser.add_argument('--optimizer', help='adam or sgd optimizer', action='store', dest='optimizer', default='sgd')
    parser.add_argument('--random_seed', help='seed for random initialization', action='store', type=int, dest='seed',
                        default=100)
    parser.add_argument('--load_model', help='load model from file', action='store_true', default=False)
    parser.add_argument('--predict', help='only predict', action='store_true', default=False)
    parser.add_argument('--unet_batch_norm', help='to choose whether use batch normalization for unet', action='store_true', default=False)
    parser.add_argument('--unet_dropout_rate', help='to set the dropout rate for unet',
                        action='store_true', default=0.5)
    parser.add_argument('--unet_channels', help='the number of unet first conv channels', action='store', default=32)

    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.debug:
        print_every = 1
        save_model_every = 10
        save_pred_every = 1
        image_size = 256
        pretrained = True
        batch_size = args.batch_size
        n_epochs = 100000
        is_validation = False
    else:
        print_every = 10
        save_model_every = 5
        save_pred_every = 2
        image_size = 224
        pretrained = True
        batch_size = args.batch_size
        n_epochs = 1000
        is_validation = False

    if args.load_model:
        print("loading model from file")
        model = torch.load('model.pt')
        print("model loaded")
    else:
        if args.model == 'vgg16fcn8':
            model = FCN8s(num_classes=args.num_classes)
        elif args.model == 'vgg16fcn16':
            model = FCN16VGG(num_classes=args.num_classes)
        elif args.model == 'vgg16fcn32':
            model = FCN32s(num_classes=args.num_classes)
        elif args.model == 'unet':
            model = UNet(3, n_classes=args.num_classes, first_conv_channels=args.unet_channels, batch_norm=args.unet_batch_norm, dropout_rate=args.unet_dropout_rate)
        else:
            raise Exception('Unknown model')
        print("Running model: " + args.model)

    cuda = torch.cuda.is_available() and args.use_gpu
    model = model.cuda() if cuda else model

    if not args.predict:
        print("training on train set")
        train_set = kagglebowl18_dataset('data',
                                         'image_train.txt',
                                         'segmentation_train.txt',
                                         'class_train.txt',
                                         image_size, validation=False, testing=False)
        val_set = kagglebowl18_dataset('data',
                                       'image_train.txt',
                                       'segmentation_train.txt',
                                       'class_train.txt',
                                       image_size, validation=True, testing=False)

        loss = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, len(val_set))
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            raise Exception('Unknown optimizer')

        trainer = Trainer(cuda=cuda,
                          model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          loss=loss,
                          optimizer=optimizer,
                          n_epochs=n_epochs,
                          n_save=save_model_every,
                          n_print=print_every,
                          learning_rate=args.learning_rate,
                          is_validation=is_validation)
        trainer.train()

    else:
        print("predicting on test set")
        val_set = kagglebowl18_dataset('data',
                                       'image_test.txt',
                                       'segmentation_test.txt',
                                       'class_test.txt',
                                       size=image_size, validation=False, testing=True)
        test_loader = DataLoader(val_set, 1)
        # for img, _, img_dir in test_loader:
        #     img = img.cuda() if cuda else img
        #     seg = model.predict(img)
        #     img = np.transpose(img.squeeze(), (1, 2, 0))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img)
        #     # plt.colorbar()
        #     # plt.show()
        #
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(seg.squeeze())
        #     # plt.colorbar()
        #     plt.show()
        submitor = Submitor(model, test_loader, output_dir='submission', cuda=cuda, threshold=20, saveseg=True)
        submitor.generate_submission_file('20180318')
