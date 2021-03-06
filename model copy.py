import torch
import torch.nn as nn
from torchvision.models import resnet50
import os
from helper import AverageMeter, save_checkpoint, accuracy


## Helper
def initialize_model(use_resnet=True, pretrained=False, nclasses=10):
    """
    
    """
    ## Initialize Model
    if use_resnet:
        model = resnet50(pretrained=pretrained)
    else:
        model = vgg16(pretrained=True)
    ## Freeze Early Layers if Pretrained
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    ## Update Output Layer
    if use_resnet:
        model.fc = nn.Linear(2048, nclasses)
    else:
        model.classifier._modules['6'] = nn.Linear(4096, nclasses)
    return model


def test_eval(val_dataloader, verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in val_dataloader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        # images = images.view(-1, 64*64)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,labels).item()

    if verbose:
        print('Test accuracy: %f %%' % (100.0 * correct / total))
        print('Test loss: %f' % (loss_sum / total))

    return 100.0 * correct / total, loss_sum / total





def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

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

    return top1.avg, top5.avg


def train(net,
          train_dataloader,
          val_dataloader,
          optimizer,
          loss_metric,
          scheduler,
          epochs = 10,
          load_model_path = None,
          save_model_name = 'resnet-tiny-10-baseline.pth',
          save_data_name = 'resnet-tiny-10-baseline_data.npz'
          ):
  
  directory_path = 'saved_model_data'
  os.makedirs(directory_path, exist_ok=True)
  if load_model_path:
    print('load trained model')
    state_dict = torch.load(load_model_path)
    net.load_state_dict(state_dict)

  #define batch train loss recording array for later visualization/plotting:
  loss_store = []
  train_perc_store = []
  test_perc_store = [] 
  test_loss_store = []
  best_test_perc = 0
  print("Starting Training")
  #training loop:
  for epoch in range(epochs):
      time1 = time.time() #timekeeping

      net.train()
      total = 0
      correct = 0
      train_loss = 0
      for i, (x,y) in enumerate(train_dataloader):

          if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

          # x = x.view(x.shape[0],-1)
          #loss calculation and gradient update:

          if i > 0 or epoch > 0:
              optimizer.zero_grad()
          outputs = net.forward(x)
          _, predicted = torch.max(outputs.data, 1)
          total += y.size(0)
          correct += (predicted.float() == y.float()).sum()
          loss = loss_metric(outputs,y)
          train_loss += loss.cpu().data.numpy().item()
          loss.backward()

          ##perform update:
          optimizer.step()

      print("Epoch",epoch+1,':')
      # train
      print('Train accuracy: %f %%' % (100.0 * correct / total))
      print('Train loss: %f' % (train_loss/total))
      loss_store.append(train_loss/total)
      train_perc = 100.0 * correct / total
      train_perc_store.append(train_perc.cpu().data.numpy().item())
      net.eval()
      # test
      test_perc, test_loss = test_eval(val_dataloader)
      test_loss_store.append(test_loss)
      test_perc_store.append(test_perc.cpu().data.numpy().item())
      # save the model with highest acc on test set
      if test_perc > best_test_perc:
        best_test_perc = test_perc
        torch.save(net.state_dict(), os.path.join(directory_path, save_model_name ))
        print('new best test acc at', epoch+1)

      scheduler.step()
      time2 = time.time() #timekeeping
      print('Elapsed time for epoch:',time2 - time1,'s')
      print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
      print()

      # save data
      save_filename = os.path.join(directory_path, save_data_name)
      np.savez(save_filename, train_perc=train_perc_store,  train_loss=loss_store, test_perc=test_perc_store, test_loss=test_loss_store)

