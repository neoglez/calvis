#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:05:21 2020

@author: neoglez
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchnet import meter
import torch.nn as nn
from CalvisDataset import CalvisFairCMU2DDataset
import visdom
import json
import datetime
import random
import platform
from recordtype import recordtype
from CalvisNet import CalvisVanillaCNNet, Calvis1CNN
from HumanBodyDimensionsTransform import TwoDToTensor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

###############################################################################
projectDir = "/home/neoglez/cmu/"
rootDir = "/home/neoglez/cmu/dataset/"
imgDir = "/home/neoglez/cmu/dataset/synthetic_images/200x200/"


opt = recordtype("Option", "env nepoch model manualSeed")
opt.env = "Calvis-training-LOOCV-200x200"
opt.nepoch = 5
opt.model = ''
opt.manualSeed = 0
opt.nb_primitives = 1
opt.num_points = 6890
opt.k_fold = 3

opt.batch_size=4
opt.num_workers=0
opt.disable_cuda = False
opt.device = 0

###############################################################################
## dataset
transformed_dataset = CalvisFairCMU2DDataset(root_dir=rootDir,
                             image_dir = imgDir,
                             transform=transforms.Compose([TwoDToTensor()]))

network = CalvisVanillaCNNet()

lrate = 0.01  # learning rate
momentum=0.6
optimizer = optim.SGD(network.parameters(), lr=lrate, momentum=momentum)
criterion = nn.L1Loss()

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/learning_with_calvis_net_experiment_1')

fold = 0
kf = KFold(n_splits=opt.k_fold)
all_indices = np.arange(len(transformed_dataset))

for train_index, test_index in kf.split(all_indices):
    fold += 1    
    
    dataset_train_sampler = SubsetRandomSampler(indices = train_index)
    dataset_test_sampler = SubsetRandomSampler(indices = test_index)
    
    trainloader = torch.utils.data.DataLoader(dataset = transformed_dataset, 
                                               batch_size = opt.batch_size, 
                                               sampler = dataset_train_sampler)
    testloader = torch.utils.data.DataLoader(dataset = transformed_dataset, 
                                              batch_size = 1,
                                              sampler = dataset_test_sampler)
    for epoch in range(1, opt.nepoch + 1):
        #train(model, optimizer, epoch, device, train_loader, log_interval)
        # Training
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):        
            # get the inputs
            inputs = data['image'].to(device=opt.device)
            labels = (data['annotations']
                ['human_dimensions']
                .to(device=opt.device))
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = network(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:    # every 10 mini-batches...
                # ...log the running loss
                writer.add_scalar('training loss',
                                running_loss / 10,
                                epoch * len(trainloader) + i)
        break

# get some random training images
#dataiter = iter(trainloader)
#nextitem = dataiter.next()
#images, labels = nextitem['image'], nextitem['annotations']['human_dimensions']

# create grid of images
#img_grid = utils.make_grid(images)


# write to tensorboard
#writer.add_image('four_synthetic_images', img_grid)
#writer.add_graph(network, images)

# helper functions

def images_to_human_dimensions(net, images):
    '''
    Generates predictions from a trained
    network and a list of images
    '''
    output = net(images)
    preds_tensor = output
    preds = np.squeeze(preds_tensor.numpy())
    return preds


def plot_human_dimension_preds(net, images, human_dimensions):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and human dimensions from a batch, that shows the network's top prediction, 
    alongside the actual human dimension, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_human_dimensions" function.
    '''
    preds = images_to_human_dimensions(net, images)
    # plot the images in the batch, along with predicted and 
    # true human dimensions
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        #matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            1.1,
            80,
            'subject'),
        color=("green" if preds[idx]==human_dimensions[idx].item() else "red"))
    return fig

# put network on GPU if we have any
network.to(device=opt.device)
