# adopted from https://github.com/jiweibo/ImageNet
import argparse
import os
import time
import numpy as np
from torchvision.models.utils import load_state_dict_from_url
from simclr import simclr_encoder
from deeplab_backbone import deeplab_backbone

from model import initialize_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from utils import TinyImageNet_data_loader, set_bn_momentum, PolyLR 
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

import deeplab_network

from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Tiny/ImageNet Training')
parser.add_argument('--dataset', default='tiny-imagenet-200-01', help='TinyImageNet or ImageNet')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('--mode', default='baseline_train', help='baseline_train/pretrain/finetune')
parser.add_argument('--pretrain_task', default='pretrain_task', help='cimclr/colorization/jigsaw')
parser.add_argument('--pretrained_model', default='', type=str, metavar='PATH', help='path to the pretrained model')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('--step_size', default=1, type=int,
                    metavar='N', help='step size (default: 1)')
parser.add_argument('--gamma', default=0.975, type=float,
                    metavar='W', help='gamma (default: 0.975)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--color_distortion', dest='color_distortion',action='store_true',
                    help='add color_distortion to train set')
parser.add_argument('-ct', '--col_test', dest='col_test',action='store_true',
                    help='try one image')
# parser.add_argument('-col', '--col', dest='col',action='store_true', type=bool,
#                     help='colorization dataloader')
parser.add_argument('--print_freq', '-f', default=40, type=int, metavar='N',
                    help='print frequency (default: 40)')
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
parser.add_argument("--num_classes", type=int, default=3,
                        help="num classes (default: 3)")
parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")


best_prec1 = 0.0
cur_itrs = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)
def main():
    global args, best_prec1
    global cur_itrs
    args = parser.parse_args()
    print(args.mode)

    # STEP1: model
    if args.mode=='baseline_train':
        model = initialize_model(use_resnet=True, pretrained=False, nclasses=200)
    elif args.mode=='pretrain':
        model = deeplab_network.deeplabv3_resnet50(num_classes=args.num_classes, output_stride=args.output_stride, pretrained_backbone=False)
        set_bn_momentum(model.backbone, momentum=0.01)
    elif args.mode=='finetune':
        # parser.add_argument('--pretrain_task', default='pretrain_task', help='simclr/colorization/jigsaw')
        if args.pretrain_task=='simclr' and args.pretrained_model:
            print("=> loading pretrained model '{}'".format(args.pretrained_model))
            model = simclr_encoder(args.pretrained_model)
            print("=> loaded pretrained model " + args.pretrain_task)
        elif args.pretrain_task=='colorization' and args.pretrained_model:
            print("=> loading pretrained model '{}'".format(args.pretrained_model))
            model = deeplab_backbone(args.pretrained_model)
            print("=> loaded pretrained model " + args.pretrain_task)

    if torch.cuda.is_available:
        model = model.cuda()
    
    # STEP2: criterion and optimizer
    if args.mode in ['baseline_train', 'finetune']:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) 
    elif args.mode=='pretrain':
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = PolyLR(optimizer, args.total_itrs, power=0.9)

    # STEP3: loss/prec record
    if args.mode in ['baseline_train', 'finetune']:
        train_losses = []
        train_top1s = []
        train_top5s = []

        test_losses = []
        test_top1s = []
        test_top5s = []
    elif args.mode == 'pretrain':
        train_losses = []
        test_losses = []

    # STEP4: optionlly resume from a checkpoint
    if args.resume:
        print('resume')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.mode in ['baseline_train', 'finetune']:
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                datafile = args.resume.split('.pth')[0] + '.npz'
                load_data = np.load(datafile)
                train_losses = list(load_data['train_losses'])
                train_top1s = list(load_data['train_top1s'])
                train_top5s = list(load_data['train_top5s'])
                test_losses = list(load_data['test_losses'])
                test_top1s = list(load_data['test_top1s'])
                test_top5s = list(load_data['test_top5s'])
            elif args.mode=='pretrain':
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                cur_itrs = checkpoint['cur_itrs']
                datafile = args.resume.split('.pth')[0] + '.npz'
                load_data = np.load(datafile)
                train_losses = list(load_data['train_losses'])
                # test_losses = list(load_data['test_losses'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # STEP5: train!
    if args.mode in ['baseline_train', 'finetune']:
        # data
        # from utils import TinyImageNet_data_loader
        print('color_distortion:', args.color_distortion)
        train_loader, val_loader = TinyImageNet_data_loader(args.dataset, args.batch_size,color_distortion=args.color_distortion)
        
        # if evaluate the model
        if args.evaluate:
            print('evaluate this model on validation dataset')
            validate(val_loader, model, criterion, args.print_freq)
            return
        
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr)
            time1 = time.time() #timekeeping

            # train for one epoch
            model.train()
            loss, top1, top5 = train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
            train_losses.append(loss)
            train_top1s.append(top1)
            train_top5s.append(top5)

            # evaluate on validation set
            model.eval()
            loss, prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)
            test_losses.append(loss)
            test_top1s.append(prec1)
            test_top5s.append(prec5)

            # remember the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_model_name = args.mode + '_' + args.pretrain_task + '_' + args.dataset +'.pth'
            if args.mode=='finetune' else args.mode + '_' + args.dataset +'.pth'
            save_checkpoint({
                'epoch': epoch + 1,
                'mode': args.mode,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, save_model_name)

            np.savez(args.mode + '_' + args.dataset +'.npz', train_losses=train_losses,train_top1s=train_top1s,train_top5s=train_top5s, test_losses=test_losses,test_top1s=test_top1s, test_top5s=test_top5s)
           # np.savez(args.mode + '_' + args.dataset +'.npz', train_losses=train_losses)
            time2 = time.time() #timekeeping
            print('Elapsed time for epoch:',time2 - time1,'s')
            print('ETA of completion:',(time2 - time1)*(args.epochs - epoch - 1)/60,'minutes')
            print()
    elif args.mode=='pretrain':
        if args.col_test:
            args.dataset = 'tiny-imagenet-200-0002'
            train_loader, val_loader = TinyImageNet_data_loader(args.dataset, 16, col=True)
            # fetch frist input and target
            epoch = 0
            while True:
                epoch+=1
                time1 = time.time()
                for gray_img, col_img in tqdm(train_loader):
                    cur_itrs+=1
                    # break
                    gray_img = gray_img.to(device, dtype=torch.float32)
                    col_img = col_img.to(device, dtype=torch.float32)

                    col_img = transforms.Resize(500)(col_img)
                    # col_img = col_img.repeat(4, 1, 1, 1)
                    gray_img = transforms.Resize(500)(gray_img)
                    gray_img = gray_img.repeat(1,3,1,1)
                    model.train()                    
                    output = model(gray_img)
                    loss = criterion(output, col_img)
                    train_losses.append(loss.cpu().data.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if cur_itrs%args.print_freq==0:
                        id = cur_itrs/args.print_freq
                        simple_visulize(model, train_loader, id)
                        save_checkpoint({
                            'epoch': epoch,
                            'mode': args.mode,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),
                            "cur_itrs": cur_itrs
                        }, True, args.mode + '_test_' + args.dataset +'.pth')

                        np.savez(args.mode + '_test_' + args.dataset +'.npz', train_losses=train_losses)
                    if cur_itrs >=  args.total_itrs:
                        return
                time2 = time.time() #timekeeping
                print('Elapsed time for epoch:',time2 - time1,'s')
                print('ETA of completion:',(time2 - time1)*(args.total_itrs - cur_itrs - 1)/60,'minutes')
                print()

        else:
            #data
            # from utils import TinyImageNet_data_loader
            # args.dataset = 'tiny-imagenet-200'
            args.batch_size = 16
            train_loader, val_loader = TinyImageNet_data_loader(args.dataset, args.batch_size, col=True)
            
            # if evaluate the model, show some results
            if args.evaluate:
                print('evaluate this model on validation dataset')
                visulization(val_loader, model, args.start_epoch)
                return

            # for epoch in range(args.start_epoch, args.epochs):
            epoch = 0
            while True:
                if cur_itrs >=  args.total_itrs:
                    return
                # adjust_learning_rate(optimizer, epoch, args.lr)
                time1 = time.time() #timekeeping

                model.train()
                # train for one epoch
                loss, _, _ = train(train_loader, model, criterion, optimizer, epoch, args.print_freq, colorization=True,scheduler=scheduler)
                train_losses.append(loss)
                

                # model.eval()
                # # evaluate on validation set
                # loss, _, _ = validate(val_loader, model, criterion, args.print_freq, colorization=True)
                # test_losses.append(loss)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'mode': args.mode,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                    "cur_itrs": cur_itrs
                }, True, args.mode + '_' + args.dataset +'.pth')

                np.savez(args.mode + '_' + args.dataset +'.npz', train_losses=train_losses)
                # scheduler.step()
                time2 = time.time() #timekeeping
                print('Elapsed time for epoch:',time2 - time1,'s')
                print('ETA of completion:',(time2 - time1)*(args.total_itrs - cur_itrs - 1)/60,'minutes')
                print()
                epoch += 1




def train(train_loader, model, criterion, optimizer, epoch, print_freq, colorization=False,scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global cur_itrs

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        cur_itrs+=1
        # measure data loading time
        data_time.update(time.time() - end)
        if args.mode=='pretrain':
            target = target.to(device, dtype=torch.float32)
            input = input.to(device, dtype=torch.float32)
        else:
            target = target.cuda()
            input = input.cuda()
        # for pretrain
        if colorization:
            input = transforms.Resize(500)(input)
            target = transforms.Resize(500)(target)
            input = input.repeat(1,3,1,1)
        # for finetune 
        if args.pretrain_task=='colorization':
            input = transforms.Resize(500)(input)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if not colorization:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if not colorization:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    # return loss, top1, top5 corresponding to each epoch
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, print_freq, colorization=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
       # target = target.cuda()
       # input = input.cuda()
        if args.mode=='pretrain':
            target = target.to(device, dtype=torch.float32)
            input = input.to(device, dtype=torch.float32)
        else:
            target = target.cuda()
            input = input.cuda()

        if colorization:
            input = transforms.Resize(500)(input)
            target = transforms.Resize(500)(target)
            input = input.repeat(1,3,1,1)
        if args.pretrain_task=='colorization':
            input = transforms.Resize(500)(input)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if not colorization:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                top1.update(prec1, input.size(0))
                top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    if not colorization:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses,
            top1=top1, top5=top5))
    else:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses))
    return losses.avg, top1.avg, top5.avg

def visulization(train_loader, model, start_epoch):
    # switch to evaluate mode
    model.eval()
    os.makedirs('visulization', exist_ok=True)
    for i, (input, target) in enumerate(train_loader):
       # target = target.cuda()
       # input = input.cuda()
        input = input[0:10]
        target = target[0:10]
        input = input.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        input = input.repeat(1,3,1,1)

        input = transforms.Resize(500)(input)
        target = transforms.Resize(500)(target)

        with torch.no_grad():
            # compute output
            output = model(input)
            # output = output.cpu()
            
        fig = plt.figure(figsize=(64., 64.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 10),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        images = []
        input_img = [transforms.ToPILImage()(x) for x in input]
        target_img = [transforms.ToPILImage()(x) for x in target]
        output_img = [transforms.ToPILImage()(x) for x in output]
        images.extend(input_img)
        images.extend(target_img)
        images.extend(output_img)
        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)

        plt.savefig(os.path.join('visulization', str(start_epoch)+'.png'))


        break

def simple_visulize(model, train_loader, id):
    for gray_img, col_img in train_loader:
        break
    gray_img = gray_img.to(device, dtype=torch.float32)
    col_img = col_img.to(device, dtype=torch.float32)

    col_img = transforms.Resize(500)(col_img)
    # col_img = col_img.repeat(4, 1, 1, 1)
    gray_img = transforms.Resize(500)(gray_img)
    gray_img = gray_img.repeat(1,3,1,1)
    model.train() 
    with torch.no_grad():                   
        output = model(gray_img)

    input = gray_img
    target = col_img
    for i in range(min(input.size(0), 5)):
        plt.figure(0)
        ax = plt.subplot(131)
        img_show = transforms.ToPILImage()(input[i])
        ax.imshow(img_show)
        ax.set_title('Input')
        ax = plt.subplot(132)
        img_show = transforms.ToPILImage()(target[i])
        ax.imshow(img_show)
        ax.set_title('Ground truth')
        ax = plt.subplot(133)
        img_show = transforms.ToPILImage()(output[i])
        ax.imshow(img_show, vmin = 0)
        # ax.imshow(img_show, vmin = 0, vmax = 255)
        ax.set_title('Prediction')
        plt.savefig('simple_visulize_' + str(i) + '_' + str(id) + '.png')


if __name__ == '__main__':
    main()

