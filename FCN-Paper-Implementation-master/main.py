import argparse
import random

import torch
from torch.utils.data import DataLoader
from torch import optim

from dataset import kagglebowl18_dataset
from model.fcn16s import FCN16VGG
from model.fcn32s import FCN32s
from model.fcn8s import FCN8s
from model.unet import UNet
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run segmentation models')
    parser.add_argument('--debug', help='Debug the model', action='store_true', default=True)
    parser.add_argument('--use_gpu', help='Debug the model', action='store_true', default=True)
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
        batch_size = 1
        n_epochs = 100000
        is_validation = False
    else:
        print_every = 10
        save_model_every = 5
        save_pred_every = 2
        image_size = 224
        pretrained = True
        batch_size = 1
        n_epochs = 1000
        is_validation = False

    if args.model == 'vgg16fcn8':
        model = FCN8s(num_classes=args.num_classes)
    elif args.model == 'vgg16fcn16':
        model = FCN16VGG(num_classes=args.num_classes)
    elif args.model == 'vgg16fcn32':
        model = FCN32s(num_classes=args.num_classes)
    elif args.model == 'unet':
        model = UNet(3, n_classes=args.num_classes)
    else:
        raise Exception('Unknown model')
    print("Running model: " + args.model)

    cuda = torch.cuda.is_available() and args.use_gpu
    print('is cuda available? ', cuda)
    model = model.cuda() if cuda else model

    train_set = kagglebowl18_dataset('kaggle_train_data',
                                     'image_train.txt',
                                     'segmentation_train.txt',
                                     'class_train.txt',
                                     image_size)
    val_set = kagglebowl18_dataset('kaggle_train_data',
                                   'image_train.txt',
                                   'segmentation_train.txt',
                                   'class_train.txt',
                                   image_size)

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
