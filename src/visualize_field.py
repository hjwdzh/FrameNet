import torch
import os
os.environ['TORCH_HOME'] = './'
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader 

from model import UNet, UNet_2
from dorn import DORN
from dataset import AffineDataset, AffineTestsDataset
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import skimage.io as sio
import sys
import Render.render as render
import math
from time import time

train_dataset = AffineTestsDataset(feat=0,root='data')

sample_batched = train_dataset[35]
color = np.ascontiguousarray(np.transpose(sample_batched['image'][0:3,:,:], (1, 2, 0)).astype('float32'))
labels = np.transpose(sample_batched['label'].numpy(), (1, 2, 0))
Qx = np.ascontiguousarray(labels[:,:,0:2].astype('float32'))
Qy = np.ascontiguousarray(labels[:,:,2:4].astype('float32'))

sio.imsave('color.png', color)
render.visualizeDirection('vis.png', color, Qx, Qy)
