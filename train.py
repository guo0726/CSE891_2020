from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--trained_ssd', default=True,
                    help='trained best ssd')
parser.add_argument('--trained_ssd_weights', default='ssd300_mAP_77.43_v2.pth',
                    help='trained best ssd path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    if args.trained_ssd:
        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            
            ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
            ssd_net.add_std()
            ssd_net.load_weights(args.resume)
            net = ssd_net
            if args.cuda:
                net = torch.nn.DataParallel(ssd_net)
                cudnn.benchmark = True

            if args.cuda:
                net = net.cuda()

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
            criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                    False, args.cuda)
        else:
            ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
            print("Loading pretrined ssd weights...")
            # ssd_weights = torch.load(args.save_folder + args.trained_ssd_weights)
            ssd_net.load_ssd_weights(args.save_folder + args.trained_ssd_weights)

            # for param in ssd_net.parameters():    #freeze the parameter from SSD
            #     param.requires_grad = False

            ssd_net.add_std()  #TODO
            net = ssd_net
            if args.cuda:
                net = torch.nn.DataParallel(ssd_net)
                cudnn.benchmark = True

            if args.cuda:
                net = net.cuda()

            if args.resume:
                print('Resuming training, loading {}...'.format(args.resume))
                ssd_net.load_weights(args.resume)

            if not args.resume:
                print('Initializing weights...')
                # initialize newly added layers' weights with xavier method
                ssd_net.std.apply(weights_init)

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
            criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                    False, args.cuda)

            
    else:
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net

        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            ssd_net.load_weights(args.resume)

        else:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

        if args.cuda:
            net = net.cuda()




        if not args.resume:
            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.std.apply(weights_init)
            ssd_net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                False, args.cuda) 

    # loss counters
    loc_loss = 0
    KL1_loss = 0
    KL2_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'KL1 Loss', 'KL2 Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    result_file = open('training_results.txt', 'w')
    result_file.close()
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, KL1_loss, KL2_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            KL1_loss = 0
            KL2_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        # load train data
        # images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        # print('kkk')
        loss_l, KL_loss_1, KL_loss_2, loss_c = criterion(out, targets)
        # print('2loss_l:', loss_l)
        # print('2kloss1:', KL_loss_1)
        # print('2kloss2:', KL_loss_2)
        # print('2loss_c:', loss_c)
        loss = loss_l + KL_loss_1 + KL_loss_2 + loss_c
        # loss = KL_loss_1 + KL_loss_2 + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        KL1_loss += KL_loss_1.data
        KL2_loss += KL_loss_2.data
        conf_loss += loss_c.data
#########################

        
        # print(iteration)
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || loss_l: %.4f ||' % (loss_l), end=' ')
            print('iter ' + repr(iteration) + ' || KL_loss_1: %.4f ||' % (KL_loss_1), end=' ')
            print('iter ' + repr(iteration) + ' || KL_loss_2: %.4f ||' % (KL_loss_2), end=' ')
            print('iter ' + repr(iteration) + ' || loss_c: %.4f ||' % (loss_c), end=' ')
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_l + KL_loss_1 + KL_loss_2 + loss_c), end=' ')
            result_file = open('training_results.txt', 'a')
            result_file.write(','.join([str(loss.item()),str(loss_l.item()), str(KL_loss_1.item()), str(KL_loss_2.item()), str(loss_c.item())])+'\n')
            result_file.close()

        if args.visdom:
            update_vis_plot(iteration, loss_l, KL_loss_1, KL_loss_2, loss_c,
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC2007_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')
    result_file.close()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def std_weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, k1, k2, conf, window1, window2, window3, window4, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, k1, k2, conf, loc + k1 + k2 + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 5)).cpu(),
            Y=torch.Tensor([loc, k1, k2, conf, loc + k1 + k2 + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
