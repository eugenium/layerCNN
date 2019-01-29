import argparse
import os
import shutil
import time
from collections import OrderedDict
import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model_greedy import *


import numpy as np




from random import randint
import datetime

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')


parser.add_argument('--ncnn',  default=8,type=int, help='depth of the auxiliary CNN')
parser.add_argument('--bn',  default=1,type=int, help='turn off/on batchnorm')
parser.add_argument('--nepochs',  default=45,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=20,type=int, help='interval between lr decay')
parser.add_argument('--avg_size',  default=7,type=int, help='size of the averaging')
parser.add_argument('--feature_size',  default=128,type=int, help='width')
parser.add_argument('--nlin',  default=2,type=int, help='number of non linearities in the auxillary')
parser.add_argument('--ds',  default=2,type=int, help='initial downsampling')
parser.add_argument('--ensemble', default=1,type=int,help='ensemble') # not implemented yet
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--prog',  default=0,type=int, help='increase width of auxillary at downsampling')
parser.add_argument('--debug', default=0,type=int,help='debugging')
parser.add_argument('--large_size_images', default=2,type=int,help='use small images for faster testing')
parser.add_argument('--n_resume', default=0,type=int,help='which layer we resume')
parser.add_argument('--resume_epoch', default=0,type=int,help='which epoch we resume')
parser.add_argument('--fixed_feat', default=512,type=int,help='auxillary width ')
parser.add_argument('--down', default=1,type=int,help='use downsampling')
parser.add_argument('--save_folder', default='.',type=str,help='down')

args = parser.parse_args()
best_prec1 = 0

time_stamp = str(datetime.datetime.now().isoformat())

name_log_txt = time_stamp + str(randint(0, 1000)) + args.name

name_log_txt=name_log_txt +'.log'

args.ensemble = args.ensemble>0
args.prog = args.prog >0
args.debug = args.debug > 0
args.bn = args.bn > 0

downsample = [1,3,5,7]

args.down = args.down > 0
def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.large_size_images==0:
        N_img = 112
        N_img_scale = 128
        print('using 112')
    elif args.large_size_images==1:
        N_img = 160
        N_img_scale = 182
        print('using 160')
    elif args.large_size_images ==2:
        N_img = 224
        N_img_scale= 256

    in_size = N_img // args.ds

    with open(name_log_txt, "a") as text_file:
        print(args, file=text_file)

    n_cnn = args.ncnn

    model = greedyNet(block_conv, 1, feature_size=args.feature_size, downsampling=args.ds,
                      downsample=downsample, batchnorm=args.bn)


    if args.fixed_feat:
        num_feat = args.fixed_feat
    else:
        num_feat = args.feature_size
    model_c = auxillary_classifier(avg_size=args.avg_size, in_size=N_img // args.ds,
                                   n_lin=args.nlin, feature_size=num_feat,
                                   input_features=args.feature_size, batchn=args.bn, num_classes=1000)

    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
        print(model_c, file=text_file)


    model = torch.nn.DataParallel(nn.Sequential(model,model_c)).cuda()
    model.module[0].unfreezeGradient(0)

    model_c = None



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    to_train = list(filter(lambda p: p.requires_grad, model.parameters())) #+ list(model_c.parameters())
    optimizer = torch.optim.SGD(to_train, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(N_img),

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))




    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(N_img_scale),
            transforms.CenterCrop(N_img),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    if (args.resume):
        def load_model():
            model_dict = torch.load(args.save_folder+'/'+args.resume + '_model.t7')

            for key in list(model_dict.keys()):
                if key[0:8]=='module.1':
                    model_dict.pop(key,None)
                else:
                    model_dict = OrderedDict((key[9:] if k == key else k, v) for k, v in model_dict.items())

            # 1. filter out unnecessary keys
            sub_dict = {k: v for k, v in model.module[0].items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(sub_dict)
            model.module[0].load_state_dict(sub_dict)
        load_model()

    correct_all = np.zeros(len(train_dataset))


    num_ep = args.nepochs



    for n in range(n_cnn):
      #  torch.save(model_c.state_dict(), name_log_txt + '_model_c.t7')
        model.module[0].unfreezeGradient(n)
        lr = args.lr * 10.0

        for epoch in range(0, num_ep):
            if n > 0 and not args.debug and epoch % 3==0:
                torch.save(model.state_dict(), args.save_folder+'/'+name_log_txt + '_current_model.t7')
            if epoch % args.epochdecay == 0:
                lr = lr/10.0
                to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
                optimizer = torch.optim.SGD(to_train, lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            if (args.resume and args.resume_epoch>0 and n==args.n_resume):
                if epoch < args.resume_epoch:
                    continue
                if epoch == args.resume_epoch:
                    name = args.resume + '_current_model.t7'
                    model_dict = torch.load(name)
                    model.load_state_dict(model_dict)
            if (args.resume and n<args.n_resume):
                if args.ensemble:
                    load_model()
                    name = args.resume  + '_' + str(n) + '_model.t7'
                    model_c_dict = torch.load(args.save_folder+'/'+name)
                    model.module[1].load_state_dict(model_c_dict)
                    top1train = -1
                    top5train = -1
                else:
                    break
            else:
                top1train, top5train= train(train_loader, model,  criterion, optimizer, epoch, n)


            # evaluate on validation set
            top1test, top5test, top1ens,top5ens = validate(val_loader, model,  criterion, n)

            prec1 = top1test

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train top1:{}(top5:{}), test top1:{} (top5:{}), top1ens:{} top5ens:{}"
                      .format(n, epoch, top1train, top5train, top1test,top5test,top1ens,top5ens), file=text_file)
            if (args.resume and n<args.n_resume ):
                break


        if args.debug:
            num_ep = 1
        else:
            torch.save(model.state_dict(), args.save_folder+'/'+name_log_txt  +'_model.t7')
            torch.save(model.module[1].state_dict(),args.save_folder+'/'+ name_log_txt  +'_'+str(n)+'_model.t7')


        if args.down and n in downsample:
            args.avg_size = int(args.avg_size / 2)
            in_size = int(in_size / 2)
            args.feature_size = int(args.feature_size * 2)
            args.fixed_feat=args.fixed_feat*2

        if args.fixed_feat:
            num_feat = args.fixed_feat
        else:
            num_feat = args.feature_size
        print('reinit classifier')
        model_c = None
        model_c = auxillary_classifier(avg_size=args.avg_size, in_size=in_size,
                                       n_lin=args.nlin, feature_size=num_feat,
                                       input_features=args.feature_size, batchn=args.bn, num_classes=1000).cuda()
        model.module[0].add_block(n in downsample)
        model = torch.nn.DataParallel(nn.Sequential(model.module[0], model_c)).cuda()
        with open(name_log_txt, "a") as text_file:
            print(model, file=text_file)

def train(train_loader, model, criterion, optimizer, epoch, n):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for k in range(n):
        model.module[0].blocks[k].eval()
        print('woof')

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output


        output = model([input_var, n])
       # output = model_c.forward(output)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.debug and i>500:
            break

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return top1.avg,top5.avg


all_outs = [[] for i in range(args.ncnn)]
def validate(val_loader, model, criterion, n):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targs = []
    # switch to evaluate mode
    model.eval()
  #  model_c.eval()

    end = time.time()
    all_outs[n] = []

    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model([input_var, n])


            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            if args.ensemble:
                all_outs[n].append(F.softmax(output).data.cpu())
                all_targs.append(target_var.data.cpu())
            total += input_var.size(0)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if args.ensemble:
        all_outs[n] = torch.cat(all_outs[n])
        all_targs = torch.cat(all_targs)
        #This is all on cpu so we dont care

        weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
        total_out = torch.zeros([total, 1000])

        # very lazy
        for i in range(n + 1):
            total_out += float(weight[i]) * all_outs[i]

        prec1, prec5 = accuracy(total_out, all_targs, topk=(1, 5))

        print(' * Ensemble Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
              .format(top1=prec1[0], top5=prec5[0]))
        return top1.avg,top5.avg,prec1[0],prec5[0]
    return top1.avg, top5.avg,-1,-1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
