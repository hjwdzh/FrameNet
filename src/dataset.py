import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import random
import skimage.io as sio
import skimage.transform as tr
import pickle
import numpy as np
import scipy.misc as misc
import os


class AffineDataset(Dataset):
    def __init__(self, root='/orion/downloads/framenet', usage='test', feat=0):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pickle.load(open(self.root + '/train_test_split.pkl', 'rb'))[usage]

        self.idx = [i for i in range(0,len(self.data_info[0]), 1)]
        self.data_len = len(self.data_info[0])

        self.intrinsics = [577.591,318.905,578.73,242.684]
        xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
        self.mesh_x = misc.imresize((xx - self.intrinsics[1]) / self.intrinsics[0], (240,320),'nearest',mode='F')
        self.mesh_y = misc.imresize((yy - self.intrinsics[3]) / self.intrinsics[2], (240,320),'nearest',mode='F')
        self.feat = feat
        self.root = root + '/scannet-frames'

    def __getitem__(self, index):
        # Get image name from the pandas df
        color_info = self.data_info[0][self.idx[index]]
        orient_info = self.data_info[1][self.idx[index]]
        orient_info_X = self.data_info[1][self.idx[index]][:-10] + 'orient-X.png'
        orient_info_Y = self.data_info[1][self.idx[index]][:-10] + 'orient-Y.png'
        mask_info = self.data_info[2][self.idx[index]]

        color_info = self.root + '/' + color_info[26:]
        orient_info = self.root + '/' + orient_info[27:]
        orient_info_X = self.root + '/' + orient_info_X[27:]
        orient_info_Y = self.root + '/' + orient_info_Y[27:]
        mask_info = self.root + '/' + mask_info[27:]
        orient_mask_tensor = misc.imresize(sio.imread(mask_info), (240,320), 'nearest')

        # Open image
        color_img = misc.imresize(sio.imread(color_info), (240,320,3), 'nearest')
        color_tensor = self.to_tensor(color_img)
        input_tensor = np.zeros((5, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3,:,:] = color_tensor
        input_tensor[3,:,:] = self.mesh_x
        input_tensor[4,:,:] = self.mesh_y

        if self.feat == 1:
            depth_info = color_info[:-9] + 'renderdepth.png'
            depth_img = misc.imresize(sio.imread(depth_info)/1000.0,(240,320,3),'nearest',mode='F')

        orient_x = misc.imresize(sio.imread(orient_info_X), (240,320,3),'nearest')
        orient_x = (orient_x / 255.0 * 2.0 - 1.0).astype('float32')
        l1 = np.linalg.norm(orient_x, axis=2)
        for j in range(3):
            orient_x[:,:,j] /= (l1 + 1e-9)
        #X = self.to_tensor(orient_x.copy())
        X = torch.from_numpy(np.transpose(orient_x, (2,0,1)))
        #print(np.max(orient_x), X.max(), orient_x.shape, X.shape)
        orient_x[:,:,0] = orient_x[:,:,0] - self.mesh_x * orient_x[:,:,2]
        orient_x[:,:,1] = orient_x[:,:,1] - self.mesh_y * orient_x[:,:,2]
        if self.feat == 1:
            orient_x[:,:,0] /= (depth_img + 1e-7)
            orient_x[:,:,1] /= (depth_img + 1e-7)
        elif self.feat == 2:
            l = np.sqrt(orient_x[:,:,0]*orient_x[:,:,0]+orient_x[:,:,1]*orient_x[:,:,1]) + 1e-7
            orient_x[:,:,0] /= l
            orient_x[:,:,1] /= l

        orient_y = misc.imresize(sio.imread(orient_info_Y), (240,320,3), 'nearest')
        orient_y = (orient_y / 255.0 * 2.0 - 1.0).astype('float32')
        l2 = np.linalg.norm(orient_y, axis=2)
        for j in range(3):
            orient_y[:,:,j] /= (l2 + 1e-9)
        #Y = self.to_tensor(orient_y.copy())
        #print(np.max(orient_y), Y.max())
        Y = torch.from_numpy(np.transpose(orient_y, (2,0,1)))
        orient_y[:,:,0] = orient_y[:,:,0] - self.mesh_x * orient_y[:,:,2]
        orient_y[:,:,1] = orient_y[:,:,1] - self.mesh_y * orient_y[:,:,2]

        if self.feat == 1:
            orient_y[:,:,0] /= (depth_img + 1e-7)
            orient_y[:,:,1] /= (depth_img + 1e-7)
        elif self.feat == 2:
            l = np.sqrt(orient_y[:,:,0]*orient_y[:,:,0]+orient_y[:,:,1]*orient_y[:,:,1]) + 1e-7
            orient_y[:,:,0] /= l
            orient_y[:,:,1] /= l

        orient_img = np.zeros((orient_x.shape[0], orient_x.shape[1], 4), dtype='float32')
        orient_img[:,:,0] = orient_x[:,:,0] * (l1 > 0.5)
        orient_img[:,:,1] = orient_x[:,:,1] * (l1 > 0.5)
        orient_img[:,:,2] = orient_y[:,:,0] * (l2 > 0.5)
        orient_img[:,:,3] = orient_y[:,:,1] * (l2 > 0.5)
        #orient_img = np.concatenate((orient_x[:,:,0:2], orient_y[:,:,2]), axis=2)

        orient_img_vertical = orient_img.copy()
        orient_img_vertical[:,:,0:2] = orient_img[:,:,2:4]
        orient_img_vertical[:,:,2:4] = -orient_img[:,:,0:2]

        #orient_tensor = self.to_tensor(orient_img)
        #print(np.max(orient_img), orient_tensor.max())
        orient_tensor = torch.from_numpy(np.transpose(orient_img, (2,0,1)))
        #orient_vert_tensor = self.to_tensor(orient_img_vertical)
        orient_vert_tensor = torch.from_numpy(np.transpose(orient_img_vertical,(2,0,1)))
        #orient_mask_tensor = misc.imresize(orient_mask_tensor, (240,320), 'nearest')
        orient_mask_tensor = torch.Tensor(orient_mask_tensor  / 255.0)
        #orient_mask = np.reshape(orient_mask, (orient_mask.shape[0], orient_mask.shape[1], 1))
        #orient_mask_tensor = self.to_tensor(orient_mask)
        return {'image':input_tensor, 'label':orient_tensor, 'label_alt':orient_vert_tensor, 'mask':orient_mask_tensor, 'X':X, 'Y':Y}

    def __len__(self):
        return self.data_len

class AffineTestsDataset(Dataset):
    def __init__(self, root='/orion/downloads/framenet', usage='test', feat=0):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pickle.load(open(self.root + '/train_test_split.pkl', 'rb'))[usage]
        #print(len(self.data_info[0]))
        #self.data_info = [self.data_info[i][30000:30008] for i in range(3)]
        self.idx = [i for i in range(0,len(self.data_info[0]),200)]
        #random.shuffle(self.idx)

        # First column contains the image paths
        self.data_len = len(self.idx)
        self.intrinsics = [577.591,318.905,578.73,242.684]
        xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
        self.mesh_x = misc.imresize((xx - self.intrinsics[1]) / self.intrinsics[0], (240,320),'nearest',mode='F')
        self.mesh_y = misc.imresize((yy - self.intrinsics[3]) / self.intrinsics[2], (240,320),'nearest',mode='F')
        self.feat = feat
        self.root = root + '/scannet-frames'

    def __getitem__(self, index):
        # Get image name from the pandas df
        color_info = self.data_info[0][self.idx[index]]
        orient_info = self.data_info[1][self.idx[index]]
        orient_info_X = self.data_info[1][self.idx[index]][:-10] + 'orient-X.png'
        orient_info_Y = self.data_info[1][self.idx[index]][:-10] + 'orient-Y.png'
        mask_info = self.data_info[2][self.idx[index]]

        color_info = self.root + '/' + color_info[26:]
        orient_info = self.root + '/' + orient_info[27:]
        orient_info_X = self.root + '/' + orient_info_X[27:]
        orient_info_Y = self.root + '/' + orient_info_Y[27:]
        mask_info = self.root + '/' + mask_info[27:]
        orient_mask_tensor = misc.imresize(sio.imread(mask_info), (240,320), 'nearest')
 
        # Open image
        color_img = misc.imresize(sio.imread(color_info), (240,320,3), 'nearest')
        color_tensor = self.to_tensor(color_img)

        input_tensor = np.zeros((5, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3,:,:] = color_tensor
        input_tensor[3,:,:] = self.mesh_x
        input_tensor[4,:,:] = self.mesh_y

        #orient_img = sio.imread(orient_info)
        #orient_img = (orient_img / 255.0 * 2.0 - 1.0).astype('float32')

        if self.feat == 1:
            depth_info = color_info[:-9] + 'renderdepth.png'
            depth_img = misc.imresize(sio.imread(depth_info)/1000.0,(240,320,3),'nearest',mode='F')

        orient_x = misc.imresize(sio.imread(orient_info_X), (240,320,3),'nearest')
        orient_x = (orient_x / 255.0 * 2.0 - 1.0).astype('float32')
        l1 = np.linalg.norm(orient_x, axis=2)
        for j in range(3):
            orient_x[:,:,j] /= (l1 + 1e-9)
        X = self.to_tensor(orient_x.copy())

        orient_y = misc.imresize(sio.imread(orient_info_Y), (240,320,3), 'nearest')
        orient_y = (orient_y / 255.0 * 2.0 - 1.0).astype('float32')
        l2 = np.linalg.norm(orient_y, axis=2)
        for j in range(3):
            orient_y[:,:,j] /= (l2 + 1e-9)
        Y = self.to_tensor(orient_y.copy())
        

        orient_x[:,:,0] = orient_x[:,:,0] - self.mesh_x * orient_x[:,:,2]
        orient_x[:,:,1] = orient_x[:,:,1] - self.mesh_y * orient_x[:,:,2]
        if self.feat == 1:
            orient_x[:,:,0] /= (depth_img + 1e-7)
            orient_x[:,:,1] /= (depth_img + 1e-7)
        elif self.feat == 2:
            l = np.sqrt(orient_x[:,:,0]*orient_x[:,:,0]+orient_x[:,:,1]*orient_x[:,:,1]) + 1e-7
            orient_x[:,:,0] /= l
            orient_x[:,:,1] /= l

        orient_y[:,:,0] = orient_y[:,:,0] - self.mesh_x * orient_y[:,:,2]
        orient_y[:,:,1] = orient_y[:,:,1] - self.mesh_y * orient_y[:,:,2]

        if self.feat == 1:
            orient_y[:,:,0] /= (depth_img + 1e-7)
            orient_y[:,:,1] /= (depth_img + 1e-7)
        elif self.feat == 2:
            l = np.sqrt(orient_y[:,:,0]*orient_y[:,:,0]+orient_y[:,:,1]*orient_y[:,:,1]) + 1e-7
            orient_y[:,:,0] /= l
            orient_y[:,:,1] /= l

        orient_img = np.zeros((orient_x.shape[0], orient_x.shape[1], 4), dtype='float32')
        orient_img[:,:,0] = orient_x[:,:,0] * (l1 > 0.5)
        orient_img[:,:,1] = orient_x[:,:,1] * (l1 > 0.5)
        orient_img[:,:,2] = orient_y[:,:,0] * (l2 > 0.5)
        orient_img[:,:,3] = orient_y[:,:,1] * (l2 > 0.5)
        #orient_img = np.concatenate((orient_x[:,:,0:2], orient_y[:,:,2]), axis=2)

        orient_img_vertical = orient_img.copy()
        orient_img_vertical[:,:,0:2] = orient_img[:,:,2:4]
        orient_img_vertical[:,:,2:4] = -orient_img[:,:,0:2]

        orient_tensor = self.to_tensor(orient_img)
        orient_vert_tensor = self.to_tensor(orient_img_vertical)

        #orient_mask_tensor = misc.imresize(orient_mask_tensor, (240,320), 'nearest')
        orient_mask_tensor = torch.Tensor(orient_mask_tensor  / 255.0)
        #orient_mask = np.reshape(orient_mask, (orient_mask.shape[0], orient_mask.shape[1], 1))
        #orient_mask_tensor = self.to_tensor(orient_mask)
        return {'image':input_tensor, 'label':orient_tensor, 'label_alt':orient_vert_tensor, 'mask':orient_mask_tensor, 'X':X, 'Y':Y}

    def __len__(self):
        return self.data_len
