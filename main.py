'''Train ESFC with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import os
import argparse

from models import *
from utils import progress_bar
from logger import Logger, savefig

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset  # For custom datasets
import random
from models.utils import mask2dist, mask2randDist, permute_2d_to_1d, coo_diagonal, extractor, MS_MST
from models.loss import auto_correlation, KDLoss
from models.dataloader import dataloader


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--training_epoch', default=80, type=int, help='training epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='FMNIST', type=str, help='dataset')
args = parser.parse_args()

device = 'cuda'

# torch.cuda.manual_seed_all(3006)
# Data
trainset, testset, trainloader, testloader = dataloader(args=args)

# Model
print('==> Building model..')
net = resnet.resnet18()
net = net.to(device)
net_assist = resnet.resnet18_assist()
net_assist = net_assist.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net_assist = torch.nn.DataParallel(net_assist)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.checkpoint + '/best.pth')
    net.load_state_dict(checkpoint['net'])
    net_assist.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    logger = Logger(os.path.join('./log',args.checkpoint, 'log.txt'), title='', resume=True)
else:
    if not os.path.isdir(os.path.join('./log',args.checkpoint)):
        os.mkdir(os.path.join('./log',args.checkpoint))
    logger = Logger(os.path.join('./log',args.checkpoint, 'log.txt'), title='')
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc.'])

criterion_CE = nn.CrossEntropyLoss().to(device)
criterion = auto_correlation().to(device)
L2_Loss = nn.MSELoss().to(device)
kdloss = KDLoss().to(device)

optimizer = optim.Adam(
                       net.parameters(),
                       lr=args.lr,
                       betas = (0.9, 0.999),
                       eps = 1e-08,
                       weight_decay = 1e-4,
                        )

optimizer2 = optim.Adam(
                       net_assist.parameters(),
                       lr=args.lr,
                       betas = (0.9, 0.999),
                       eps = 1e-08,
                       weight_decay = 1e-4,
                        )


scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20,40,60], 0.5)

# Training
def train(epoch, auto_corr_per_batch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    net_assist.train()
    train_loss = 0
    correct = 0
    total = 0 
    loss_avg = 0
    acc_avg = 0
    ref_avg = 0
    i = 0
    for batch_idx, (images, targets) in enumerate(trainloader):
        images, targets = images.to(device), targets.to(device)

        gnn, mask2, mask3, mask4 = net(images)
        optimizer.zero_grad()

        _, _, mst2 = mask2dist(mask2, images.size()[2] // 2)
        _, _, mst3 = mask2dist(mask3, images.size()[2] // 2)
        _, _, mst4 = mask2dist(mask4, images.size()[2] // 2)
        p_coord = MS_MST(mst2, mst3, mst4, images.size()[2] // 2)

        pt = p_coord.permute(1,0).contiguous()
        b = permute_2d_to_1d(images[:,0,:,:].unsqueeze(dim=1), pt.long())
        main_score = criterion(b)

        noise = torch.rand(images.size(),device='cuda')
        gnn_assist, mask_assist2, mask_assist3, mask_assist4 = net_assist(images + noise)
        optimizer2.zero_grad()

        _, _, mst2 = mask2dist(mask_assist2, images.size()[2] // 2)
        _, _, mst3 = mask2dist(mask_assist3, images.size()[2] // 2)
        _, _, mst4 = mask2dist(mask_assist4, images.size()[2] // 2)
        p_coord = MS_MST(mst2, mst3, mst4, images.size()[2] // 2)

        pt = p_coord.permute(1, 0).contiguous()
        b = permute_2d_to_1d(images[:, 0, :, :].unsqueeze(dim=1), pt.long())
        assis_score = criterion(b)

        if assis_score >= main_score:
            loss = kdloss(gnn, gnn_assist)
            loss.backward()
            optimizer.step()
        else:
            loss = kdloss(gnn_assist, gnn)
            loss.backward()
            optimizer2.step()

        train_loss += loss.item()
        total += targets.size(0)
        loss_avg = train_loss / (batch_idx + 1)
        acc_avg += torch.max(main_score, assis_score)
        i += 1

        progress_bar(batch_idx, len(trainloader), 'AC: %.3f%% '
            % (acc_avg / (batch_idx + 1)*100))

    return (loss_avg, acc_avg)


def test(epoch):
    global best_acc
    net.eval()
    net_assist.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_avg = 0
    acc_avg = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            images, targets = images.to(device), targets.to(device)

            gnn, mask = net(images)

            p_coord = mask2dist(mask, images.size()[2] // 2)[1]
            pt = p_coord.permute(1, 0).contiguous()
            b = permute_2d_to_1d(images[:, 0, :, :].unsqueeze(dim=1), pt.long())
            main_score = criterion(b)

            acc_avg += main_score

            progress_bar(batch_idx, len(testloader), 'AC: %.3f%%'
                % (acc_avg / (batch_idx + 1)*100))

    acc = acc_avg / (batch_idx + 1)*100
    if acc > best_acc:
        print('Saving Best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint/' + args.checkpoint):
            os.mkdir('./checkpoint/' + args.checkpoint )
        torch.save(state,'./checkpoint/' + args.checkpoint + '/best.pth')

        state = {
            'net': net_assist.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint/' + args.checkpoint):
            os.mkdir('./checkpoint/' + args.checkpoint )
        torch.save(state,'./checkpoint/' + args.checkpoint + '/best_assist.pth')

        best_acc = acc

    print('Best acc: %.2f' % best_acc)
    return (loss_avg, acc_avg, best_acc)


if __name__ == '__main__':

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    auto_corr_per_batch = np.zeros((1, len(trainset)//args.batch_size+1))

    for epoch in range(start_epoch, start_epoch+args.training_epoch):
        # scheduler2.step()
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.training_epoch, scheduler.get_lr()[0]))
        train_loss, train_acc = train(epoch, auto_corr_per_batch)
        test_loss, test_acc, best_acc = test(epoch)
        scheduler.step()
        # append logger file
        logger.append([scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc, best_acc])
        # savefig(os.path.join('./log', args.checkpoint, 'log.pdf'))
        # logger.plot()
    logger.close()