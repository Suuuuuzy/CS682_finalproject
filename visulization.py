def main(val_loader, model, criterion, print_freq, colorization=False):
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
        input = input.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        if colorization:
            input = input.repeat(1,3,1,1)

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

if __name__ == '__main__':
    main()
