#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:05:21 2020

@author: neoglez
The overall structure was inspired by "Visualizing Models, Data, and Training
with TensorBoard"
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
opt.nepoch = 1
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
## Helper functions
###############################################################################
# helper function to show an image
# (used in the `plot_human_dimension_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().detach().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def images_to_human_dimensions(net, images):
    '''
    Generates predictions from a trained
    network and a list of images
    '''
    output = net(images)
    # just for readability
    preds_tensor = output
    preds = preds_tensor.cpu().detach().numpy()
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
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(("ground truth:\n"
                      "chest circumference: {0:.2f}cm,\n"
                      "waist circumference: {1:.2f}cm,\n"
                      "pelvis circumference: {2:.2f}cm,\n"
                      "estimated:\n"
                      "chest circumference: {3:.2f}cm,\n"
                      "waist circumference: {4:.2f}cm,\n"
                      "pelvis circumference: {5:.2f}cm").format(
            preds[0][0], preds[0][1], preds[0][2],
            preds[1][0], preds[1][1], preds[1][2]
            ), color="green")
    return fig

def write_one_batch_to_tensorboard(writer, images, bodydimensions=None):
    # create grid of images
    img_grid = utils.make_grid(images)
    # write to tensorboard
    writer.add_image('four_synthetic_images', img_grid)
    
###############################################################################
## dataset
transformed_dataset = CalvisFairCMU2DDataset(root_dir=rootDir,
                             image_dir = imgDir,
                             transform=transforms.Compose([TwoDToTensor()]))

network = CalvisVanillaCNNet()
network.to(device=opt.device)

lrate = 0.01  # learning rate
momentum=0.6
optimizer = optim.SGD(network.parameters(), lr=lrate, momentum=momentum)
criterion = nn.L1Loss()

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/learning_with_calvis_net_experiment_1')

fold = 0
kf = KFold(n_splits=opt.k_fold)
all_indices = np.arange(len(transformed_dataset))

showbatch = True

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
        # Training
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  
            # just once
            if showbatch:
                write_one_batch_to_tensorboard(writer, data['image'])
                showbatch = False
            
            # get the inputs
            inputs = data['image'].to(device=opt.device)
            # get only the information that we need and stream out the 
            # batch dimension
            labels = (data['annotations']
                ['human_dimensions'][:, -1, :]
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
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                        plot_human_dimension_preds(network, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
        break


# write to tensorboard
#writer.add_image('four_synthetic_images', img_grid)
#writer.add_graph(network, images)
