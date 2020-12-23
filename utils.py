import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random
import imagesize
import cv2
from pprint import pprint
from numba import jit
from IPython.display import display

import torch
from torch._C import device
from torch.utils.data import Dataset


class Root:
    def  __init__(self, rootDir):
        self.rootDir = rootDir
    
    def __call__(self, filename):
        return os.path.join(self.rootDir, filename)


class ImageDataset(Dataset):
    def __init__(self, labelDataFrame, isLabel, imageNameColumn, imageLabelColumn='', imageDir='', format='.png', transform=None):
        super().__init__()
        self.labelDf   = labelDataFrame.reset_index()
        self.isLabel   = isLabel
        self.nameCol   = imageNameColumn
        self.labelCol  = imageLabelColumn
        self.imageDir  = imageDir
        self.format    = format
        self.transform = transform
    
    def __getitem__(self, i):
        img   = cv2.imread(self.GetImagePath(i))[:, :, ::-1].copy()
        if self.transform:
            img = self.transform(img)

        if self.isLabel:
            label = self.labelDf[self.labelCol][i]
            return img, label
        else:
            return img
    
    def __len__(self):
        return self.labelDf.shape[0]
    
    def GetImagePath(self, i):
        filename = self.labelDf[self.nameCol][i] + self.format
        return os.path.join(self.imageDir, filename)


@jit
def PlotImageSizeHist(dirs, bins=100, xRight=1000, figsize=(18, 10)):
    widths, heights = [], []
    for dir in dirs:
        paths = glob.glob(os.path.join(dir, '*'))
        for path in paths:
            width, height = imagesize.get(path)
            widths .append(width)
            heights.append(height)
    
    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.hist(widths,  bins); plt.title("Width") ; plt.xlim([0, xRight]); plt.xticks(range(0, xRight+1, xRight//bins), rotation=60)
    plt.subplot(212)
    plt.hist(heights, bins); plt.title("Height"); plt.xlim([0, xRight]); plt.xticks(range(0, xRight+1, xRight//bins), rotation=60)
    return widths, heights


def GPUToNumpy(tensor, reduceDim=None):
    if reduceDim is not None:
        return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
    else:
        return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)


def BuildImagePathsDataFrame(imageDir, columnName='id_code', format='', isKeepDir=False):
    paths = glob.glob(os.path.join(imageDir, '*' + format))
    if isKeepDir:
        return pd.DataFrame({
            columnName: paths
        })
    else:
        df = pd.DataFrame({
            columnName: paths
        })

        df[columnName] = df[columnName].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        return df


def ToOneHot(labels, dim=5):
    oneHot = torch.zeros([labels.shape[0], dim]).to(labels.device)
    oneHot.scatter_(1, labels.view(oneHot.shape[0], 1), 1.)
    return oneHot


def GetAccuracy(pred, real):
    real = real.to(pred.device)
    if pred.shape != real.shape:
        pred = torch.argmax(pred, dim=1)

    return torch.sum(pred == real) / pred.shape[0]


def CheckAllRequiresGrad(model):
    require = True
    for p in model.parameters():
        require = require and p.requires_grad
    
    return require


def SetLR(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr


def PlotLRSchedule(scheduler, iterations):
    for i in range(iterations):
        plt.plot(i, scheduler.get_last_lr(), 'bo')
        scheduler.step()


def SaveState(model, optimizer, scheduler, epoch, path):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, path)
    return state


def LoadState(model, optimizer, scheduler, path):
    state = torch.load(path)
    epoch = state['epoch']
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    return epoch, model, optimizer, scheduler