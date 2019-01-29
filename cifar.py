"Greedy layerwise cifar training"
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model_greedy import *
from torch.autograd import Variable

from utils import progress_bar

from random import randint
import datetime
import json



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ncnn',  default=5,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=15,type=int, help='number of epochs')
parser.add_argument('--avg_size',  default=16,type=int, help='size of averaging ')
parser.add_argument('--feature_size',  default=128,type=int, help='feature size')
parser.add_argument('--ds-type', default=None, help="type of downsampling. Defaults to old block_conv with psi. Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--nlin',  default=2,type=int, help='nlin')
parser.add_argument('--ensemble', default=1,type=int,help='compute ensemble')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--batch_size', default=128,type=int,help='batch size')
parser.add_argument('--bn', default=0,type=int,help='use batchnorm')
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='verification that layers frozen')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--width_aux', default=128,type=int,help='auxillary width')
parser.add_argument('--down', default='[1,2]', type=str,
                        help='layer at which to downsample')
parser.add_argument('--seed', default=None, help="Fixes the CPU and GPU random seeds to a specified number")

args = parser.parse_args()
opts = vars(args)
args.ensemble = args.ensemble>0
args.bn = args.bn > 0
args.debug_parameters = args.debug_parameters > 0

if args.debug:
    args.nepochs = 1 # we run just one epoch per greedy layer training in debug mode

downsample =  list(np.array(json.loads(args.down)))
in_size=32
mode=0

if args.seed is not None:
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


save_name = 'layersize_'+str(args.feature_size)+'width_' \
            + str(args.width_aux) + 'depth_' + str(args.nlin) + 'ds_type_' + str(args.ds_type) +'down_' +  args.down 
#logging
time_stamp = str(datetime.datetime.now().isoformat())

name_log_dir = ''.join('{}{}-'.format(key, val) for key, val in sorted(opts.items()))+time_stamp
name_log_dir = 'runs/'+name_log_dir

name_log_txt = time_stamp + save_name + str(randint(0, 1000)) + args.name
debug_log_txt = name_log_txt + '_debug.log'
name_save_model = name_log_txt + '.t7'
name_log_txt=name_log_txt   +'.log'

with open(name_log_txt, "a") as text_file:
    print(opts, file=text_file)


use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset_class = torchvision.datasets.CIFAR10(root='.', train=True, download=True,transform=transform_train)
trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# Model

print('==> Building model..')
n_cnn=args.ncnn
#if args.ds_type is None:
 #   block_conv_ = block_conv
#else:
 #   from functools import partial
  #  block_conv_ = partial(ds_conv, ds_type=args.ds_type)
net = greedyNet(block_conv, 1, args.feature_size, downsample=downsample, batchnorm=args.bn)
    

if args.width_aux:
    num_feat = args.width_aux
else:
    num_feat = args.feature_size

net_c = auxillary_classifier(avg_size=args.avg_size, in_size=in_size,
                             n_lin=args.nlin, feature_size=num_feat,
                             input_features=args.feature_size, batchn=args.bn)


with open(name_log_txt, "a") as text_file:
    print(net, file=text_file)
    print(net_c, file=text_file)

net = torch.nn.DataParallel(nn.Sequential(net,net_c)).cuda()
cudnn.benchmark = True

criterion_classifier = nn.CrossEntropyLoss()

net.module[0].unfreezeGradient(0)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


def train_classifier(epoch,n):
    print('\nSubepoch: %d' % epoch)
    net.train()
    for k in range(n):
        net.module[0].blocks[k].eval()


    if args.debug_parameters:
        #This is used to verify that early layers arent updated
        import copy
        #store all parameters on cpu as numpy array
        net_cpu = copy.deepcopy(net).cpu()
        net_cpu_dict = net_cpu.module[0].state_dict()
        with open(debug_log_txt, "a") as text_file:
            print('n: %d'%n)
            for param in net_cpu_dict.keys():
                net_cpu_dict[param]=net_cpu_dict[param].numpy()
                print("parameter stored on cpu as numpy: %s  "%(param),file=text_file)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net.forward([inputs,n])

        loss = criterion_classifier(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_pers=0

        if args.debug_parameters:

            s_dict = net.module[0].state_dict()
            with open(debug_log_txt, "a") as text_file:
                print("iteration %d" % (batch_idx), file=text_file)
                for param in s_dict.keys():
                    diff = np.sum((s_dict[param].cpu().numpy()-net_cpu_dict[param])**2)
                    print("n: %d parameter: %s size: %s changed by %.5f" % (n,param,net_cpu_dict[param].shape,diff),file=text_file)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  losspers: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,loss_pers))

    acc = 100.*float(correct)/float(total)
    return acc

all_outs = [[] for i in range(args.ncnn)]

def test(epoch,n,ensemble=False):
    global acc_test_ensemble
    all_targs = []
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_outs[n] = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net([inputs,n])

        loss = criterion_classifier(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

        if args.ensemble:
            all_outs[n].append(outputs.data.cpu())
            all_targs.append(targets.data.cpu())
    acc = 100. * float(correct) / float(total)

    if ensemble:
        all_outs[n] = torch.cat(all_outs[n])
        all_targs = torch.cat(all_targs)
        #This is all on cpu so we dont care
        weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
        total_out = torch.zeros((total,10))

        #very lazy
        for i in range(n+1):
            total_out += float(weight[i])*all_outs[i]


        _, predicted = torch.max(total_out, 1)
        correct = predicted.eq(all_targs).sum()
        acc_ensemble = 100*float(correct)/float(total)
        print('Acc_ensemble: %.2f'%acc_ensemble)
    if ensemble:
        return acc,acc_ensemble
    else:
        return acc

i=0
num_ep = args.nepochs

for n in range(n_cnn):
    net.module[0].unfreezeGradient(n)
    lr = args.lr*5.0# we run at epoch 0 the lr reset to remove non learnable param

    for epoch in range(0, num_ep):
        i=i+1
        print('n: ',n)
        if epoch % args.epochdecay == 0:
            lr=lr/5.0
            to_train = list(filter(lambda p: p.requires_grad, net.parameters()))
            optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=5e-4)
            print('new lr:',lr)

        acc_train = train_classifier(epoch,n)
        if args.ensemble:
            acc_test,acc_test_ensemble = test(epoch,n,args.ensemble)

            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train {}, test {},ense {} "
                      .format(n,epoch,acc_train,acc_test,acc_test_ensemble), file=text_file)
        else:
            acc_test = test(epoch, n)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train {}, test {}, ".format(n,epoch,acc_train,acc_test), file=text_file)

        if args.debug:
            break


    if args.down and n in downsample:
        args.avg_size = int(args.avg_size/2)
        in_size = int(in_size/2)
        args.feature_size = int(args.feature_size*2)
        args.width_aux = args.width_aux * 2

    if args.width_aux:
        num_feat = args.width_aux
    else:
        num_feat = args.feature_size

    net_c = None
    if n < n_cnn-1:
        net_c = auxillary_classifier(avg_size=args.avg_size, in_size=in_size,
                                     n_lin=args.nlin, feature_size=args.width_aux,
                                     input_features=args.feature_size, batchn=args.bn).cuda()
        net.module[0].add_block(n in downsample)
        net = torch.nn.DataParallel(nn.Sequential(net.module[0], net_c)).cuda()

state_final = {
            'net': net,
            'acc_test': acc_test,
            'acc_train': acc_train,
        }
torch.save(state_final,save_name)
