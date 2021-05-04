import argparse
from model import initialize_model, test_eval, train
import torch

parser = argparse.ArgumentParser(description='PyTorch Tiny/ImageNet Training')
parser.add_argument('--dataset', default='TinyImageNet', help='TinyImageNet or ImageNet')
parser.add_argument('--mode', default='baseline_train', help='baseline_train/pretrain/finetune')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
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


def main():
    global args
    args = parser.parse_args()
    print(args.mode)
    # model
    if args.mode=='baseline_train':
        save_model_name = 'resnet-tiny-10-baseline.pth',
        save_data_name = 'resnet-tiny-10-baseline_data.npz'
        model = initialize_model(use_resnet=True, pretrained=False, nclasses=200)
    if torch.cuda.is_available:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) 
    loss_metric = torch.nn.CrossEntropyLoss()
    # data
    if args.dataset=='TinyImageNet':
        from utils import TinyImageNet_data_loaders
        train_dataloader, val_dataloader = TinyImageNet_data_loaders(args.batch-size)
    # train
    train(model,
          train_dataloader,
          val_dataloader,
          optimizer,
          loss_metric,
          train_scheduler,
          epochs = args.epochs,
          save_model_name = save_model_name,
          save_data_name = save_data_name
          )


if __name__ == '__main__':
    main()

