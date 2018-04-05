import argparse

import torchvision

from dataset.reader import *
from dataset.transform import *
from net.configuration import Configuration
from net.model import MaskNet
from utility.file import Logger
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description='Script to run segmentation models')
    parser.add_argument('--batch_size', help='desired batch size for training', action='store', type=int,
                        dest='batch_size', default=1)
    parser.add_argument('--net', help='models to train on', action='store', dest='models', default='mask_rcnn')
    parser.add_argument('--learning_rate', help='starting learning rate', action='store', type=float,
                        dest='learning_rate', default=0.001)
    parser.add_argument('--optimizer', help='adam or sgd optimizer', action='store', dest='optimizer', default='sgd')
    parser.add_argument('--random_seed', help='seed for random initialization', action='store', type=int, dest='seed',
                        default=100)
    parser.add_argument('--load_model', help='load models from file', action='store_true', default=False)
    parser.add_argument('--predict', help='only predict', action='store_true', default=False)
    parser.add_argument('--print_every', help='print loss every print_every steps', action='store', type=int,
                        default=10)
    parser.add_argument('--save_model_every', help='save models every save_model_every steps', action='store', type=int,
                        default=100)
    parser.add_argument('--input_width', help='input image width to a net', action='store', type=int, default=128)
    parser.add_argument('--input_height', help='input image height to a net', action='store', type=int, default=128)
    parser.add_argument('--pretrained', help='load pretrained models when doing transfer learning', action='store_true',
                        default=True)
    parser.add_argument('--num_epochs', help='total number of epochs for training', action='store', type=int,
                        default=100000)
    parser.add_argument('--is_validation', help='whether or not calculate validation when training',
                        action='store_true', default=False)
    parser.add_argument('--iter_valid', help='calculate validation loss every validation_every steps',
                        action='store', type=int, default=100)
    parser.add_argument('--num_workers', help='number of workers for loading dataset', action='store', type=int,
                        default=4)
    parser.add_argument('--train_split', help='the train dataset split', choices=[
        'train1_ids_all_670',
        'train1_ids_gray2_500',
        'debug1_ids_gray2_10',
        'disk0_ids_dummy_9',
        'disk0_ids_dummy_10',
        'purple_108',
        'train1_ids_purple_only1_101',
        'merge1_1'], action='store', default='train1_ids_gray2_500')
    parser.add_argument('--valid_split', help='the train dataset split', choices=[
        'valid1_ids_gray2_43',
        'debug1_ids_gray2_10',
        'disk0_ids_dummy_3',
        'train1_ids_purple_only1_101',
        'merge1_1'], action='store', default='valid1_ids_gray2_43')
    parser.add_argument('--iter_accum', help='iter_accum', action='store', type=int, default=1)
    parser.add_argument('--result_dir', help='result dir for saving logs and data', action='store', default='results')
    parser.add_argument('--data_dir', help='the root dir to store data', action='store', default='../data')
    parser.add_argument('--initial_checkpoint', help='check point to load model', action='store', default=None)
    parser.add_argument('--image_folder_train', help='the folder containing images for training', action='store',
                        default='stage1_train')
    parser.add_argument('--image_folder_valid', help='the folder containing images for validation', action='store',
                        default='stage1_train')
    parser.add_argument('--image_folder_test', help='the folder containing images for testing', action='store',
                        default='stage1_test')
    parser.add_argument('--masks_folder_train', help='the folder containing masks for training', action='store',
                        default='fixed_multi_masks')
    parser.add_argument('--masks_folder_valid', help='the folder containing masks for validation', action='store',
                        default='fixed_multi_masks')

    args = parser.parse_args()

    # debug
    if 1:
        args.batch_size = 1
        args.print_every = 1
        args.learning_rate = 0.002
        args.iter_valid = 1
        args.is_validation = False
        args.train_split = 'purple_108'
        args.input_width = 128
        args.input_height = 128
        args.iter_accum = 1
        args.seed = 0
        args.num_workers = 1
        args.save_model_every = 1000

    os.makedirs(args.result_dir, exist_ok=True)
    print('data_dir', args.data_dir)
    print('result_dir', args.result_dir)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    log = Logger()
    log.open(args.result_dir + '/log.train.txt', mode='a')

    # net ----------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskNet(cfg, 3)
    net = net.cuda() if USE_CUDA else net

    log.write('** dataset setting **\n')

    WIDTH, HEIGHT = args.input_width, args.input_height

    def train_augment(image, multi_mask, meta, index):
        image, multi_mask = random_shift_scale_rotate_transform2(image, multi_mask,
                                                                 shift_limit=[0, 0], scale_limit=[1 / 2, 2],
                                                                 rotate_limit=[-45, 45],
                                                                 borderMode=cv2.BORDER_REFLECT_101,
                                                                 u=0.5)  # borderMode=cv2.BORDER_CONSTANT

        # overlay = multi_mask_to_color_overlay(multi_mask,color='cool')
        # overlay1 = multi_mask_to_color_overlay(multi_mask1,color='cool')
        # image_show('overlay',overlay)
        # image_show('overlay1',overlay1)
        # cv2.waitKey(0)

        image, multi_mask = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT, u=0.5)
        image, multi_mask = random_horizontal_flip_transform2(image, multi_mask, 0.5)
        image, multi_mask = random_vertical_flip_transform2(image, multi_mask, 0.5)
        image, multi_mask = random_rotate90_transform2(image, multi_mask, 0.5)
        # image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)
        # ---------------------------------------
        H, W = image.shape[0], image.shape[1]
        input = torch.from_numpy(image.reshape(H, W, -1).transpose((2, 0, 1))).float().div(255)

        # TODO add padding to image???
        box, label, instance = multi_mask_to_annotation(multi_mask)

        return input, box, label, instance, meta, index

    def valid_augment(image, multi_mask, meta, index):
        image, multi_mask = fix_crop_transform2(image, multi_mask, -1, -1, WIDTH, HEIGHT)

        # ---------------------------------------
        input = torch.from_numpy(image[..., np.newaxis].transpose((2, 0, 1))).float().div(255)
        box, label, instance = multi_mask_to_annotation(multi_mask)

        return input, box, label, instance, meta, index

    def train_collate(batch):
        batch_size = len(batch)
        inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
        boxes = [batch[b][1] for b in range(batch_size)]
        labels = [batch[b][2] for b in range(batch_size)]
        instances = [batch[b][3] for b in range(batch_size)]
        metas = [batch[b][4] for b in range(batch_size)]
        indices = [batch[b][5] for b in range(batch_size)]

        return [inputs, boxes, labels, instances, metas, indices]

    train_dataset = ScienceDataset(
        data_dir=args.data_dir,
        image_set=args.train_split,
        image_folder=args.image_folder_train,
        masks_folder=args.masks_folder_train,
        color_scheme=cv2.IMREAD_COLOR,
        transform=train_augment, mode='train')

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collate)

    valid_dataset = ScienceDataset(
        data_dir=args.data_dir,
        image_set=args.valid_split,
        image_folder=args.image_folder_valid,
        masks_folder=args.masks_folder_valid,
        color_scheme=cv2.IMREAD_COLOR,
        transform=valid_augment, mode='valid')

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n' % (WIDTH, HEIGHT))
    log.write('\ttrain_dataset.split = %s\n' % train_dataset.image_set)
    log.write('\tvalid_dataset.split = %s\n' % valid_dataset.image_set)
    log.write('\tlen(train_dataset)  = %d\n' % (len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n' % (len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n' % (len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n' % (len(valid_loader)))
    log.write('\tbatch_size  = %d\n' % args.batch_size)
    log.write('\titer_accum  = %d\n' % args.iter_accum)
    log.write('\tbatch_size*iter_accum  = %d\n' % (args.batch_size * args.iter_accum))
    log.write('\n')

    LR = None  # LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=args.learning_rate / args.iter_accum, momentum=0.9, weight_decay=0.0001)

    trainer = Trainer(net=net, train_loader=train_loader, val_loader=valid_loader, optimizer=optimizer,
                      learning_rate=args.learning_rate, LR=LR, logger=log,
                      iter_accum=args.iter_accum, num_iters=1000 * 1000,
                      iter_smooth=20, iter_log=args.print_every, iter_valid=args.iter_valid,
                      images_per_epoch=len(train_dataset),
                      initial_checkpoint=args.initial_checkpoint, pretrain_file=None, debug=True,
                      is_validation=args.is_validation,
                      out_dir=args.result_dir)

    trainer.run_train()


if __name__ == '__main__':
    main()
    # # resnet18 = torchvision.models.resnet18(pretrained=True)
    # # print(resnet18)
    # #
    # # print('####################################################')
    # cfg = Configuration()
    # net = MaskNet(cfg, 1)
    # print(net.state_dict().keys())
    # # print(net)
    # #
    # # print('####################################################')
    # resnet50 = torchvision.models.resnet50(pretrained=True)
    # # print(resnet50)
    # print(resnet50.state_dict().keys())
