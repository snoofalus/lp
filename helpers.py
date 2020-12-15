import re
import argparse
import os
import shutil
import time
import math

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

import lp.db_semisuper as db_semisuper

'''
Short migration guide:
model = MyRNN()
if use_cuda:
    model = model.cuda() 
NOW
device = torch.device("cuda" if use_cuda else "cpu")
model = MyRNN().to(device)

generally tensor.data NOW tensor.detach() 
No longer need for variables, previous volatile variables NOW goes under an with torch.no_grad():

'''

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    # LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print("--- checkpoint copied to %s ---" % best_path)

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)

    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size, args.fully_supervised])

###
#Numpy 180+CH images
###
#----------------------------------------------------------------------------------------------------
    if args.dataset=='s1s2glcm16' or args.dataset=='s1s2glcm8' or args.dataset=='s18' or args.dataset=='s28' or args.dataset=='glcm8':

        def npy_loader(path):
            #sample = torch.from_numpy(np.load(path))
            sample = np.load(path)
            return sample

        #dataset = db_semisuper.DBSS(root=traindir, transform=train_transformation, loader=npy_loader)
        dataset = db_semisuper.DBSS(root=traindir, transform=train_transformation, loader=npy_loader)
    
        if not args.fully_supervised and args.labels:
            with open(args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

        if args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.fully_supervised:
            sampler = SubsetRandomSampler(range(len(dataset)))
            dataset.labeled_idx = range(len(dataset))
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=0,#args.workers
                                                   pin_memory=True)

        train_loader_noshuff = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers= 0,  # Needs images twice as fast args.workers
            pin_memory=True,
            drop_last=False)

        eval_dataset = torchvision.datasets.DatasetFolder(
            root=evaldir,
            loader=npy_loader,
            transform=eval_transformation,
            extensions=('.npy')
        )


        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0, #args.workers
            pin_memory=True,
            drop_last=False)


        return train_loader, eval_loader, train_loader_noshuff, dataset

###
#Regular image propagation
###
#----------------------------------------------------------------------------------------------------

    else:
        dataset = db_semisuper.DBSS(traindir, train_transformation) # torchvision imagefolder object in original paper, here custom made

        if not args.fully_supervised and args.labels:
            with open(args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

        if args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.fully_supervised:
            sampler = SubsetRandomSampler(range(len(dataset)))
            dataset.labeled_idx = range(len(dataset))
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=0,#args.workers
                                                   pin_memory=True)

        train_loader_noshuff = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers= 0,  # Needs images twice as fast args.workers
            pin_memory=True,
            drop_last=False)

        eval_dataset = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0, #args.workers
            pin_memory=True,
            drop_last=False)

        return train_loader, eval_loader, train_loader_noshuff, dataset


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     torch.mul(ema_param.data, alpha)
    #     torch.mul((1-alpha), param.data)
    #     torch.add(ema_param.data, param.data)

    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(num_classes,args,ema=False):
    model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=num_classes, isL2 = args.isL2, double_output = args.double_output)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).to(torch.device("cuda:0"))

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def train(train_loader, model, optimizer, epoch, global_step, args, ema_model = None):
    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none').to(torch.device("cuda:0"))
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    if ema_model is not None:
        isMT = True
    else:
        isMT = False

    # switch to train mode
    model.train()
    if isMT:
        ema_model.train()

    end = time.time()

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):

        if isMT:
            images = batch_input[0]
            ema_images = batch_input[1]
        else:
            images = batch_input


        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        meters.update('lr', optimizer.param_groups[0]['lr'])

        target_var = target.to(torch.device("cuda:0"), non_blocking=True)
        #print(target_var)
        weight_var = weight.to(torch.device("cuda:0"), non_blocking=True)
        c_weight_var = c_weight.to(torch.device("cuda:0"), non_blocking=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.detach().ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if isMT: 
            #print(ema_images.requires_grad) #false
            #print(ema_images.shape) #59,6
            ema_logit, _ , _ = ema_model(ema_images)
            class_logit, cons_logit, _ = model(images)

            ema_logit = ema_logit.detach()

            if args.logit_distance_cost >= 0:
                res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.item())
            else:
                res_loss = 0

            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch,args)
                meters.update('cons_weight', consistency_weight)
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)


        else:
            class_logit, cons_logit = model(images)

        loss = class_criterion(class_logit, target_var) 
        loss = loss * weight_var.float()
        loss = loss * c_weight_var
        loss = loss.sum() / minibatch_size
        meters.update('class_loss', loss.item())

        if isMT:    
            loss = loss + consistency_loss + res_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        #tensors dont have a data attribute, variables do. just use tensor instead of tensor.data
        prec1, prec3 = accuracy(class_logit.detach(), target_var.detach(), topk=(1, 3))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top3', prec3[0], labeled_minibatch_size)
        meters.update('error3', 100. - prec3[0], labeled_minibatch_size)

        if isMT:
            ema_prec1, ema_prec3 = accuracy(ema_logit, target_var, topk=(1, 3))
            meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_top3', ema_prec3[0], labeled_minibatch_size)
            meters.update('ema_error3', 100. - ema_prec3[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if isMT:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]'
                'LR {meters[lr]:.4f}\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@3 {meters[top3]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))


    return meters , global_step

#variables and tensors are the same as of pt 0.4 and higher, use with 
def validate(eval_loader, model, global_step, epoch, isMT = False):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).to(torch.device("cuda:0"))
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval() # turns off dropout/batchnorm, not related to gradients

    end = time.time()
    with torch.no_grad(): # no gradients for less mem usage
        for i, (images, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            target = target.to(torch.device("cuda:0"))
            minibatch_size = list(target.size())[0] # int 50
            labeled_minibatch_size = target.ne(NO_LABEL).sum() #tensor 50
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            if isMT:
                output1, _, _ = model(images)
            else:
                output1, _ = model(images)
            class_loss = class_criterion(output1, target) / minibatch_size # should be single float number

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output1.detach(), target.detach(), topk=(1, 3))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1[0], labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
            meters.update('top3', prec3[0], labeled_minibatch_size)
            meters.update('error3', 100.0 - prec3[0], labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@3 {top3.avg:.3f}'
          .format(top1=meters['top1'], top3=meters['top3']))

    #return meters['top1'].avg, meters['top3'].avg
    return meters


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:

        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].flatten(start_dim=0).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res

def extract_features(train_loader,model, isMT = False):
    model.eval()
    embeddings_all, labels_all, index_all = [], [], []

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:
            X = batch_input[0]
        else:
            X = batch_input

        y = batch_input[1]

        X = X.to(torch.device("cuda:0"))
        y = y.to(torch.device("cuda:0"), non_blocking=True)

        if isMT:
            _ , _ , feats = model(X)
        else:
            _ , feats = model(X)

        embeddings_all.append(feats.detach().cpu())
        labels_all.append(y.detach().cpu())

    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    labels_all = torch.cat(labels_all).numpy()
    return (embeddings_all, labels_all)

def load_args(args, isMT = False):

    label_dir = '../data-local/'

    if args.dataset == "cifar100":
        args.batch_size = 128
        args.lr = 0.2
        args.test_batch_size = args.batch_size

        args.epochs = 180
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)
        args.arch = 'cifar_cnn'

    elif args.dataset == "cifar10":

        args.test_batch_size = args.batch_size
        args.epochs = 10
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.arch = 'cifar_cnn'

        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)
        #print(args.labels)
        #exit()

    elif args.dataset == "miniimagenet":

        args.train_subdir = 'train'
        args.evaluation_epochs = 30

        args.epochs = 180
        args.batch_size = 128
        args.lr = 0.2
        args.test_batch_size = args.batch_size

        args.epochs = 180
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)
        args.arch = 'resnet18'

    elif args.dataset == "s1s2glcm16":

        args.test_batch_size = args.batch_size
        args.epochs = 150
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.arch = 's1s2glcm_CNN16'

        args.labels = '%s/labels/s1s2glcm/%d_balanced_labels/%d.txt' % (label_dir,args.num_labeled,args.label_split)

    elif args.dataset == "s1s2glcm8":

        args.test_batch_size = args.batch_size
        args.epochs = 80
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        #args.arch = 's1s2glcm_RES'
        args.arch = 's1s2glcm_CNN8'

        args.labels = '%s/labels/s1s2glcm/%d_balanced_labels/%d.txt' % (label_dir,args.num_labeled,args.label_split)

    elif args.dataset == "s18":

        args.test_batch_size = args.batch_size
        args.epochs = 80
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        #args.arch = 's1_RES'
        args.arch = 's1_CNN8'

        args.labels = '%s/labels/s1s2glcm/%d_balanced_labels/%d.txt' % (label_dir,args.num_labeled,args.label_split)

    elif args.dataset == "s28":

        args.test_batch_size = args.batch_size
        args.epochs = 80
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        #args.arch = 's2_RES'
        args.arch = 's2_CNN8'

        args.labels = '%s/labels/s1s2glcm/%d_balanced_labels/%d.txt' % (label_dir,args.num_labeled,args.label_split)

    elif args.dataset == "glcm8":

        args.test_batch_size = args.batch_size
        args.epochs = 80
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        #args.arch = 'glcm_RES'
        args.arch = 'glcm_CNN8'

        args.labels = '%s/labels/s1s2glcm/%d_balanced_labels/%d.txt' % (label_dir,args.num_labeled,args.label_split)

    else:
        sys.exit('Undefined dataset!')

    if isMT:
        args.double_output = True
    else:
        args.double_output = False

    return args