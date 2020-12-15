# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import torchvision.transforms as transforms
import numpy as np

from . import data
from .utils import export

import os
import pdb

@export
def cifar10(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))


    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]

    data_dir = '../data-local/images/cifar/cifar10/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar100(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616]) # should we use different stats - do this
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/cifar/cifar100/by-image'

    print("Using CIFAR-100 from", data_dir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }


@export
def miniimagenet(isTwice=True):
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))


    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/miniimagenet'
    

    print("Using mini-imagenet from", data_dir)


    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }


@export
def s1s2glcm16(isTwice=True):
    norm = np.load(os.path.abspath('../data-local/workdir/norm.npy'))
    mu = norm[:,0].tolist()
    standard = norm[:,1].tolist()

    channel_stats = dict(mean=mu,
                         std=standard)

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            #data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(**channel_stats)
        ]))


    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            #data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]

    data_dir = '../data-local/images/s1s2glcm/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 6
    }

@export
def s1s2glcm8(isTwice=True):
    norm = np.load(os.path.abspath('../data-local/workdir/norm.npy'))
    mu = norm[:,0].tolist()
    standard = norm[:,1].tolist()

    channel_stats = dict(mean=mu,
                         std=standard)

    
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomPadandCrop(32),
            data.RandomFlip(),
            data.ToTensor(),
            transforms.Normalize(**channel_stats),
        ]))

    else:
        train_transformation = transforms.Compose([
            data.RandomPadandCrop(32),
            data.RandomFlip(),
            data.ToTensor(),
            transforms.Normalize(**channel_stats),
        ])

    eval_transformation = transforms.Compose([
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])


    myhost = os.uname()[1]

    data_dir = '../data-local/images/s1s2glcm/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 6
    }

@export
def s18(isTwice=True):
    norm = np.load(os.path.abspath('../data-local/workdir/norm.npy'))
    mu = norm[169::13,0].tolist()
    standard = norm[169::13,1].tolist()

    channel_stats = dict(mean=mu,
                         std=standard)

    train_transformation = transforms.Compose([
        data.RandomPadandCrop(32),
        data.RandomFlip(),
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    eval_transformation = transforms.Compose([
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    myhost = os.uname()[1]

    data_dir = '../data-local/images/s1/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 6
    }

@export
def s28(isTwice=True):
    norm = np.load(os.path.abspath('../data-local/workdir/norm.npy'))
    mu = norm[0:169:13,0].tolist()
    standard = norm[0:169:13,1].tolist()

    channel_stats = dict(mean=mu,
                         std=standard)

    train_transformation = transforms.Compose([
        data.RandomPadandCrop(32),
        data.RandomFlip(),
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    eval_transformation = transforms.Compose([
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    myhost = os.uname()[1]

    data_dir = '../data-local/images/s2/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 6
    }

@export
def glcm8(isTwice=True):
    norm = np.load(os.path.abspath('../data-local/workdir/norm.npy'))
    dd=np.arange(0,180,12)
    norm = np.delete(norm, dd, axis=0)
    mu = norm[:,0].tolist()
    standard = norm[:,1].tolist()

    channel_stats = dict(mean=mu,
                         std=standard)

    train_transformation = transforms.Compose([
        data.RandomPadandCrop(32),
        data.RandomFlip(),
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    eval_transformation = transforms.Compose([
        data.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    myhost = os.uname()[1]

    data_dir = '../data-local/images/glcm/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 6
    }


