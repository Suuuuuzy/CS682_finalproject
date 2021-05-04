# adopted from https://github.com/jiweibo/ImageNet
import argparse
import os
import time
import numpy as np

from model import initialize_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from utils import TinyImageNet_data_loader
from helper import AverageMeter, save_checkpoint, accuracy

parser = argparse.ArgumentParser(description='PyTorch Tiny/ImageNet Training')
parser.add_argument('--dataset', default='TinyImageNet', help='TinyImageNet or ImageNet')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('--mode', default='baseline_train', help='baseline_train/pretrain/finetune')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('--step_size', default=1, type=int,
                    metavar='N', help='step size (default: 1)')
parser.add_argument('--gamma', default=0.975, type=float,
                    metavar='W', help='gamma (default: 0.975)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--print_freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')

best_prec1 = 0.0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args.mode)

    # model
    if args.mode=='baseline_train':
        model = initialize_model(use_resnet=True, pretrained=False, nclasses=200)
    if torch.cuda.is_available:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) 
    criterion = nn.CrossEntropyLoss()

    # optionlly resume from a checkpoint
    if args.resume:
        print('resume')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # data
    if args.dataset=='TinyImageNet':
        from utils import TinyImageNet_data_loader
        train_loader, val_loader = TinyImageNet_data_loader(args.batch_size)
    
    # if evaluate the model
    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return
    
    train_losses = []
    train_top1s = []
    train_top5s = []

    test_losses = []
    test_top1s = []
    test_top5s = []

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr)
        time1 = time.time() #timekeeping

        # train for one epoch
        loss, top1, top5 = train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
        train_losses.append(loss)
        train_top1s.append(top1)
        train_top5s.append(top5)

        # evaluate on validation set
        loss, prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)
        test_losses.append(loss)
        test_top1s.append(prec1)
        test_top5s.append(prec5)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'mode': args.mode,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.mode + '_' + args.dataset +'.pth')

        np.savez(args.mode + '_' + args.dataset +'.npz', train_losses=train_losses,train_top1s=train_top1s,train_top5s=train_top5s, test_losses=test_losses,test_top1s=test_top1s, test_top5s=test_top5s)
        train_scheduler.step()
       # np.savez(args.mode + '_' + args.dataset +'.npz', train_losses=train_losses)
        time2 = time.time() #timekeeping
        print('Elapsed time for epoch:',time2 - time1,'s')
        print('ETA of completion:',(time2 - time1)*(args.epochs - epoch - 1)/60,'minutes')
        print()



def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    # return loss, top1, top5 corresponding to each epoch
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()

