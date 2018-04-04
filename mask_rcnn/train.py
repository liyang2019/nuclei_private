from net.rate import *
from net.draw import *
from net.model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'


# -------------------------------------------------------------------------------------
class Trainer:

    def __init__(self, net, train_loader, val_loader, optimizer, learning_rate, LR, logger,
                 iter_accum, num_iters, iter_smooth, iter_log, iter_valid, images_per_epoch,
                 initial_checkpoint, pretrain_file, debug, is_validation, out_dir):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.iter_accum = iter_accum
        self.num_iters = num_iters
        self.iter_smooth = iter_smooth
        self.iter_log = iter_log
        self.iter_valid = iter_valid
        self.initial_checkpoint = initial_checkpoint
        self.pretrain_file = pretrain_file
        self.log = logger
        self.debug = debug
        self.LR = LR
        self.images_per_epoch = images_per_epoch
        self.is_validation = is_validation
        self.out_dir = out_dir

    def run_train(self):

        # pretrain_file = None  # RESULTS_DIR + '/mask-single-shot-dummy-1a/checkpoint/00028000_model.pth'
        skip = ['crop', 'mask']

        # setup  -----------------
        os.makedirs(os.path.join(self.out_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join('../backup'), exist_ok=True)
        # backup_project_as_zip(PROJECT_PATH, os.path.join('../backup/maskrcnn.code.train.%s.zip' % IDENTIFIER))

        self.log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        self.log.write('** some experiment setting **\n')
        self.log.write('\tSEED         = %u\n' % SEED)
        self.log.write('\tout_dir      = %s\n' % self.out_dir)
        self.log.write('\n')

        cfg = self.net.cfg

        print('initial_checkpoint', self.initial_checkpoint)
        if self.initial_checkpoint is not None:
            print('\tinitial_checkpoint = %s\n' % self.initial_checkpoint)
            self.log.write('\tinitial_checkpoint = %s\n' % self.initial_checkpoint)
            self.net.load_state_dict(torch.load(self.initial_checkpoint, map_location=lambda storage, loc: storage))

        if self.pretrain_file is not None:
            self.log.write('\tpretrain_file = %s\n' % self.pretrain_file)
            self.net.load_pretrain(self.pretrain_file, skip)

        self.log.write('%s\n' % self.net.version)
        self.log.write('\n')

        iter_save = [0, self.num_iters - 1] + list(range(0, self.num_iters, 500))  # 1*1000

        start_iter = 0
        start_epoch = 0.
        if self.initial_checkpoint is not None:
            checkpoint = torch.load(self.initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']

            rate = get_learning_rate(self.optimizer)  # load all except learning rate
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            adjust_learning_rate(self.optimizer, rate)

        # start training here! ##############################################
        self.log.write('** start training here! **\n')
        self.log.write(' optimizer=%s\n' % str(self.optimizer))
        self.log.write(' momentum=%f\n' % self.optimizer.param_groups[0]['momentum'])
        self.log.write(' LR=%s\n\n' % str(self.LR))

        self.log.write(' images_per_epoch = %d\n\n' % self.images_per_epoch)
        self.log.write(
            ' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
        self.log.write(
            '-------------------------------------------------------------------------------------------------------------------------------\n')

        train_loss = np.zeros(6, np.float32)
        # train_acc = 0.0
        valid_loss = np.zeros(6, np.float32)
        # valid_acc = 0.0
        batch_loss = np.zeros(6, np.float32)
        # batch_acc = 0.0
        rate = 0

        start = timer()
        j = 0
        i = 0
        epoch = 0
        while i < self.num_iters:  # loop over the dataset multiple times

            sum_train_loss = np.zeros(6, np.float32)
            sum_train_acc = 0.0
            sum = 0

            self.net.set_mode('train')
            self.optimizer.zero_grad()
            for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in self.train_loader:
                if all(len(b) == 0 for b in truth_boxes):
                    continue
                batch_size = len(indices)
                i = j / self.iter_accum + start_iter
                epoch = (i - start_iter) * batch_size * self.iter_accum / self.images_per_epoch + start_epoch
                num_products = epoch * self.images_per_epoch

                if self.is_validation and i % self.iter_valid == 0:
                    self.net.set_mode('valid')
                    valid_loss, valid_acc = self.evaluate(self.net, self.val_loader)
                    self.net.set_mode('train')

                    print('\r', end='', flush=True)
                    self.log.write(
                        '%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' %
                        (
                            rate, i / 1000, epoch, num_products / 1000000,
                            valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                            # valid_acc,
                            train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                            # train_acc,
                            batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                            # batch_acc,
                            time_to_str((timer() - start) / 60)),
                        is_terminal=0)
                    time.sleep(0.01)

                # if 1:
                if i in iter_save:
                    torch.save(self.net.state_dict(), self.out_dir + '/checkpoint/%08d_model.pth' % i)
                    torch.save({
                        'optimizer': self.optimizer.state_dict(),
                        'iter': i,
                        'epoch': epoch,
                    }, self.out_dir + '/checkpoint/%08d_optimizer.pth' % i)
                    with open(self.out_dir + '/checkpoint/configuration.pkl', 'wb') as pickle_file:
                        pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

                # learning rate schduler -------------
                if self.LR is not None:
                    lr = self.LR.get_rate(i)
                    if lr < 0:
                        break
                    adjust_learning_rate(self.optimizer, lr / self.iter_accum)
                rate = get_learning_rate(self.optimizer) * self.iter_accum

                # one iteration update  -------------
                inputs = inputs.cuda() if USE_CUDA else inputs
                inputs = Variable(inputs)

                # self.net(inputs, truth_boxes, truth_labels, truth_instances)
                # loss = self.net.loss(inputs, truth_boxes, truth_labels, truth_instances)

                # TODO training rcnn only!
                self.net.forward(inputs, truth_boxes, truth_labels, truth_instances)
                # self.net.forward_train(inputs, truth_boxes, truth_labels, truth_instances)
                # loss = self.net.loss_train_rcnn(inputs, truth_boxes, truth_labels, truth_instances)
                loss = self.net.loss(inputs, truth_boxes, truth_labels, truth_instances)

                # accumulated update
                loss.backward()  # here iter_smooth does no effect
                if j % self.iter_accum == 0:
                    # torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # print statistics  ------------
                batch_acc = 0  # acc[0][0]
                batch_loss = np.array((
                    loss.cpu().data.numpy(),
                    self.net.rpn_cls_loss.cpu().data.numpy(),
                    self.net.rpn_reg_loss.cpu().data.numpy(),
                    self.net.rcnn_cls_loss.cpu().data.numpy(),
                    self.net.rcnn_reg_loss.cpu().data.numpy(),
                    # np.zeros(1, np.float32),  # TODO train rcnn only
                    self.net.mask_cls_loss.cpu().data.numpy(),
                ))
                sum_train_loss += batch_loss
                sum_train_acc += batch_acc
                sum += 1

                if i % self.iter_smooth == 0:
                    train_loss = sum_train_loss / sum
                    # train_acc = sum_train_acc / sum
                    sum_train_loss = np.zeros(6, np.float32)
                    sum_train_acc = 0.
                    sum = 0

                if i % self.iter_log == 0:
                    print(
                        '\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s \n' %
                        (
                            rate, i / 1000, epoch, num_products / 1000000,
                            valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                            # valid_acc,
                            train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                            # train_acc,
                            batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                            # batch_acc,
                            time_to_str((timer() - start) / 60), i, j, ''
                        ), end='', flush=True)  # str(inputs.size()))
                j = j + 1

                # <debug> ===================================================================
                if self.debug and i % self.iter_valid == 0:
                    # if 1:

                    self.net.set_mode('test')
                    with torch.no_grad():
                        # self.net(inputs, truth_boxes, truth_labels, truth_instances)
                        self.net.forward(inputs, truth_boxes, truth_labels, truth_instances)
                        # self.net.forward_train(inputs, truth_boxes, truth_labels, truth_instances)

                    images = inputs.data.cpu().numpy()
                    # window = self.net.rpn_window
                    # rpn_logits_flat = self.net.rpn_logits_flat.data.cpu().numpy()
                    # rpn_deltas_flat = self.net.rpn_deltas_flat.data.cpu().numpy()
                    rpn_proposals = self.net.rpn_proposals.data.cpu().numpy()

                    # rcnn_logits = self.net.rcnn_logits.data.cpu().numpy()
                    # rcnn_deltas = self.net.rcnn_deltas.data.cpu().numpy()
                    # rcnn_proposals = self.net.rcnn_proposals.data.cpu().numpy()
                    rcnn_proposals = self.net.get_detections(inputs).data.cpu().numpy()

                    # detections = self.net.detections.data.cpu().numpy()
                    # masks = self.net.masks
                    masks = self.net.get_masks(inputs)

                    # print('train',batch_size)
                    # for b in range(batch_size):
                    for b in range(1):

                        image = (images[b].transpose((1, 2, 0)) * 255)
                        image = image.astype(np.uint8)
                        image = image.repeat(3, axis=2)  # TODO for gray scale image
                        # image = np.clip(image.astype(np.float32)*2,0,255).astype(np.uint8)  #improve contrast

                        truth_box = truth_boxes[b]
                        truth_label = truth_labels[b]
                        truth_instance = truth_instances[b]
                        # truth_mask = instance_to_multi_mask(truth_instance)

                        # rpn_logit_flat = rpn_logits_flat[b]
                        # rpn_delta_flat = rpn_deltas_flat[b]
                        # rpn_prob_flat = np_softmax(rpn_logit_flat)

                        rpn_proposal = np.zeros((0, 7), np.float32)
                        if len(rpn_proposals) > 0:
                            index = np.where(rpn_proposals[:, 0] == b)[0]
                            rpn_proposal = rpn_proposals[index]

                        rcnn_proposal = np.zeros((0, 7), np.float32)
                        if len(rcnn_proposals) > 0:
                            index = np.where(rcnn_proposals[:, 0] == b)[0]
                            # rcnn_logit = rcnn_logits[index]
                            # rcnn_delta = rcnn_deltas[index]
                            # rcnn_prob = np_softmax(rcnn_logit)
                            rcnn_proposal = rcnn_proposals[index]

                        mask = masks[b]

                        # box = proposal[:,1:5]
                        # mask = masks[b]

                        # draw --------------------------------------------------------------------------
                        # contour_overlay = multi_mask_to_contour_overlay(truth_mask, image, [255,255,0] )
                        # color_overlay   = multi_mask_to_color_overlay(mask)

                        # all1 = draw_multi_rpn_prob(cfg, image, rpn_prob_flat)
                        # all2 = draw_multi_rpn_delta(cfg, overlay_contour, rpn_prob_flat, rpn_delta_flat, window,[0,0,255])
                        # all3 = draw_multi_rpn_proposal(cfg, image, proposal)
                        # all4 = draw_truth_box(cfg, image, truth_box, truth_label)

                        all5 = draw_multi_proposal_metric(cfg, image, rpn_proposal, truth_box, truth_label,
                                                          [0, 255, 255],
                                                          [255, 0, 255], [255, 255, 0])
                        all6 = draw_multi_proposal_metric(cfg, image, rcnn_proposal, truth_box, truth_label,
                                                          [0, 255, 255],
                                                          [255, 0, 255], [255, 255, 0])
                        all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

                        # image_show('color_overlay',color_overlay,1)
                        # image_show('rpn_prob',all1,1)
                        # image_show('rpn_prob',all1,1)
                        # image_show('rpn_delta',all2,1)
                        # image_show('rpn_proposal',all3,1)
                        # image_show('truth_box',all4,1)
                        # image_show('rpn_precision',all5,1)
                        # image_show('rpn_precision', all5, 1)
                        # image_show('rcnn_precision', all6, 1)
                        # image_show('mask_precision', all7, 1)

                        # summary = np.vstack([
                        #     all5,
                        #     np.hstack([
                        #         all1,
                        #         np.vstack( [all2, np.zeros((H,2*W,3),np.uint8)])
                        #     ])
                        # ])
                        # draw_shadow_text(summary, 'iter=%08d'%i,  (5,3*HEIGHT-15),0.5, (255,255,255), 1)
                        # image_show('summary',summary,1)

                        # name = train_dataset.ids[indices[b]].split('/')[-1]
                        # cv2.imwrite(out_dir + '/train/%s.rpn_precision.png' % name, all5)
                        # cv2.imwrite(out_dir + '/train/%s.rcnn_precision.png' % name, all6)

                        cv2.imwrite(self.out_dir + '/train/%05d.%02d.%s.rpn_precision.png' % (i, b, metas[b]), all5)
                        cv2.imwrite(self.out_dir + '/train/%05d.%02d.%s.rcnn_precision.png' % (i, b, metas[b]), all6)
                        cv2.imwrite(self.out_dir + '/train/%05d.%02d.%s.mask_precision.png' % (i, b, metas[b]), all7)
                        # cv2.waitKey(1)
                        pass

                    self.net.set_mode('train')
                # <debug> ===================================================================

            pass  # -- end of one data loader --
        pass  # -- end of all iterations --

        if 1:  # save last
            torch.save(self.net.state_dict(), self.out_dir + '/checkpoint/%d_model.pth' % i)
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'iter': i,
                'epoch': epoch,
            }, self.out_dir + '/checkpoint/%d_optimizer.pth' % i)

        self.log.write('\n')

    # training ##############################################################
    @staticmethod
    def evaluate(net, test_loader):
        test_num = 0
        test_loss = np.zeros(6, np.float32)
        test_acc = 0
        for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, indices) in enumerate(test_loader, 0):
            batch_size = len(indices)
            test_num += batch_size
            with torch.no_grad():
                inputs = inputs.cuda() if USE_CUDA else inputs
                inputs = Variable(inputs)
                if all(len(b) == 0 for b in truth_boxes):
                    print('all None in evaluateeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                    continue
                # net(inputs, truth_boxes, truth_labels, truth_instances)
                # loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

                # TODO train rcnn only
                net.forward(inputs, truth_boxes, truth_labels, truth_instances)
                # net.forward_train(inputs, truth_boxes, truth_labels, truth_instances)
                # loss = net.loss_train_rcnn(inputs, truth_boxes, truth_labels, truth_instances)
                loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

            # acc    = dice_loss(masks, labels) #todo
            test_acc += 0  # batch_size*acc[0][0]
            test_loss += batch_size * np.array((
                loss.cpu().data.numpy(),
                net.rpn_cls_loss.cpu().data.numpy(),
                net.rpn_reg_loss.cpu().data.numpy(),
                net.rcnn_cls_loss.cpu().data.numpy(),
                net.rcnn_reg_loss.cpu().data.numpy(),
                net.mask_cls_loss.cpu().data.numpy(),  # TODO train rcnn only
                # 0,
            ))

        assert (test_num == len(test_loader.sampler))
        test_acc = test_acc / test_num
        test_loss = test_loss / test_num
        return test_loss, test_acc


# --------------------------------------------------------------


# main #################################################################
if __name__ == '__main__':
    pass
