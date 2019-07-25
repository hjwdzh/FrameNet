import torch
import os
os.environ['TORCH_HOME'] = './'
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader 

from model import UNet
from dorn import DORN
from geonet import GeoNet
from dataset import AffineTestsDataset
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import skimage.io as sio
import scipy.misc as misc
num_epochs = 10000
batch_size = 1

parser = argparse.ArgumentParser(description='Process saome integers.')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--root', type=str, default=2)
parser.add_argument('--evaluate', type=str,default='normal')
args = parser.parse_args()

cnn = DORN(channel=5,output_channel=13)

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3);

test_dataset = AffineTestsDataset(root=args.root, feat=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
						shuffle=False, num_workers=0)

cnn = cnn.cuda()

if (args.resume != ''):
	cnn.load_state_dict(torch.load(args.resume))

def log(str):
	if args.save != '':
		fp.write('%s\n'%(str))
		fp.flush()
	#os.fsync(fp)
	print(str)

n_iter = 0
m_iter = 0

test_iter = iter(test_dataloader)

errors = None
errors0 = None
errors1 = None
errors2 = None

def ConvertToAngle(Q):
	angle1 = torch.atan2(Q[:,1:2,:,:],Q[:,0:1,:,:]) / np.pi * 180
	angle2 = torch.atan2(Q[:,3:4,:,:],Q[:,2:3,:,:]) / np.pi * 180
	angles = torch.cat([angle1, angle2], dim=1)

	q1 = torch.cos(angle1 / 180.0 * np.pi)
	q2 = torch.sin(angle1 / 180.0 * np.pi)

	return angles

def ConvertToDirection(Q):
	x1 = torch.cos(Q[:,0:1,:,:] / 180.0 * np.pi)
	y1 = torch.sin(Q[:,0:1,:,:] / 180.0 * np.pi)
	x2 = torch.cos(Q[:,1:2,:,:] / 180.0 * np.pi)
	y2 = torch.sin(Q[:,1:2,:,:] / 180.0 * np.pi)

	return torch.cat([x1,y1,x2,y2], dim=1)

def Rotate90(Q):
	Q0 = Q.clone()
	Q0[:,0,:,:] = Q[:,2,:,:]
	Q0[:,1,:,:] = Q[:,3,:,:]
	Q0[:,2,:,:] = -Q[:,0,:,:]
	Q0[:,3,:,:] = -Q[:,1,:,:]
	return Q0

def Normalize(dir_x):
	dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-7).view(dir_x.shape[0],1,dir_x.shape[2],dir_x.shape[3])
	#dir_x_l = (torch.norm(dir_x, p=2, dim=1) + 1e-7).view(dir_x.shape[0],1,dir_x.shape[2],dir_x.shape[3])
	if dir_x.shape[1] == 3:
		dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
	elif dir_x.shape[1] == 2:
		dir_x_l = torch.cat([dir_x_l, dir_x_l], dim=1)
	return dir_x / dir_x_l

def train_one_iter(i, sample_batched, evaluate=0):
	global errors, errors0, errors1, errors2
	cnn.eval()
	images = sample_batched['image']
	labels = sample_batched['label']
	masks_tensor = sample_batched['mask'] > 0
	X = sample_batched['X'].cuda()
	Y = sample_batched['Y'].cuda()

	images_tensor = Variable(images.float())
	labels_tensor = Variable(labels.float())

	images_tensor, labels_tensor = images_tensor.cuda(), labels_tensor.cuda()
	
	masks_tensor = masks_tensor.cuda()

	masks_tensor = masks_tensor.float()
	elems = torch.sum(masks_tensor).item()
	if elems == 0:
		return
	# Forward + Backward + Optimize
	optimizer.zero_grad()

	if args.evaluate == 'normal':
		outputs = cnn(images_tensor)[:,10:13,:,:]

		norm1 = Normalize(outputs)
		norm2 = Normalize(torch.cross(X,Y))
		mask = masks_tensor

		mask = mask[0].data.cpu().numpy()
		dot_product = torch.sum(norm1 * norm2, dim=1)
		dot_product = torch.clamp(dot_product,min=-1.0,max=1.0)
		angles = torch.acos(dot_product) * masks_tensor / np.pi * 180

		norm1_copy = norm1.clone()
		norm2_copy = norm2.clone()

		for j in range(3):
			norm1 = norm1_copy.clone()
			norm1[:,j,:,:] = 0
			norm2 = norm2_copy.clone()
			norm2[:,j,:,:] = 0
			norm1 = Normalize(norm1)
			norm2 = Normalize(norm2)
			dot_product = torch.sum(norm1 * norm2, dim=1)
			dot_product = torch.clamp(dot_product,min=-1.0,max=1.0)
			if j == 0:
				angles0 = torch.acos(dot_product) * masks_tensor / np.pi * 180
			elif j == 1:
				angles1 = torch.acos(dot_product) * masks_tensor / np.pi * 180
			else:
				angles2 = torch.acos(dot_product) * masks_tensor / np.pi * 180



	elif args.evaluate == 'projection':
		outputs = cnn(images_tensor)[:,0:4,:,:]
		preds = ConvertToAngle(outputs)
		mask = masks_tensor
		l0 = labels_tensor
		a1 = ConvertToAngle(l0)
		l1 = Rotate90(l0)
		a2 = ConvertToAngle(l1)
		l2 = Rotate90(l1)
		a3 = ConvertToAngle(l2)
		l3 = Rotate90(l2)
		a4 = ConvertToAngle(l3)
		d0 = preds - a1
		d0 = torch.min(torch.abs(d0), torch.min(torch.abs(d0 + 360), torch.abs(d0 - 360)))
		d0 = torch.sum(d0, dim=1)
		d1 = preds - a2
		d1 = torch.min(torch.abs(d1), torch.min(torch.abs(d1 + 360), torch.abs(d1 - 360)))
		d1 = torch.sum(d1, dim=1)
		d2 = preds - a3
		d2 = torch.min(torch.abs(d2), torch.min(torch.abs(d2 + 360), torch.abs(d2 - 360)))
		d2 = torch.sum(d2, dim=1)
		d3 = preds - a4
		d3 = torch.min(torch.abs(d3), torch.min(torch.abs(d3 + 360), torch.abs(d3 - 360)))
		d3 = torch.sum(d3, dim=1)
		d = torch.min(d0, torch.min(d1, torch.min(d2, d3)))
		d = d * mask
		angles = d / 2

	elif args.evaluate == 'principal':
		outputs = cnn(images_tensor)[:,4:10,:,:]
		dir_x = Normalize(outputs[:,0:3,:,:])
		dir_y = Normalize(outputs[:,3:6,:,:])
		X = Normalize(X)
		Y = Normalize(Y)

		angles0 = torch.acos(torch.clamp(torch.sum(dir_x*X,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180\
				+ torch.acos(torch.clamp(torch.sum(dir_y*Y,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180
		angles1 = torch.acos(torch.clamp(torch.sum(-dir_x*Y,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180\
				+ torch.acos(torch.clamp(torch.sum(dir_y*X,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180
		angles2 = torch.acos(torch.clamp(torch.sum(-dir_x*X,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180\
				+ torch.acos(torch.clamp(torch.sum(-dir_y*Y,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180
		angles3 = torch.acos(torch.clamp(torch.sum(dir_x*Y,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180\
				+ torch.acos(torch.clamp(torch.sum(-dir_y*X,dim=1), min=-1.0,max=1.0)) * masks_tensor / np.pi * 180

		angles = torch.min(angles0, torch.min(angles1, torch.min(angles2,angles3))) * 0.5
		mask1 = (angles == angles0).float()
		mask2 = (angles == angles1).float()
		mask3 = (angles == angles2).float()
		mask4 = (angles == angles3).float()

		selected_X = mask1 * X - mask2 * Y - mask3 * X + mask4 * Y
		selected_Y = mask1 * Y + mask2 * X - mask3 * Y - mask4 * X

		for j in range(3):
			dir_x_copy = dir_x.clone()
			dir_y_copy = dir_y.clone()
			X_copy = selected_X.clone()
			Y_copy = selected_Y.clone()
			X_copy[:,j,:,:] = 0
			Y_copy[:,j,:,:] = 0
			dir_x_copy[:,j,:,:] = 0
			dir_y_copy[:,j,:,:] = 0
			X_copy = Normalize(X_copy)
			Y_copy = Normalize(Y_copy)
			dir_x_copy = Normalize(dir_x_copy)
			dir_y_copy = Normalize(dir_y_copy)

			a = torch.acos(torch.clamp(torch.sum(dir_x_copy*X_copy,dim=1),min=-1.0,max=1.0)) * masks_tensor / np.pi * 180\
					+ torch.acos(torch.clamp(torch.sum(dir_y_copy*Y_copy,dim=1),min=-1.0,max=1.0)) * masks_tensor / np.pi * 180

			if j == 0:
				angles0 = a
			elif j == 1:
				angles1 = a
			else:
				angles2 = a


	masks_np = masks_tensor.data.cpu().numpy() > 0

	angles_np = angles.data.cpu().numpy()
	angles_np = angles_np[masks_np]
	if errors is None:
		errors = angles_np.copy()
	else:
		errors = np.concatenate((errors, angles_np))

	if i % 10 == 0 or i > 320:
		print('Item %d of %d: Mean %f, Median %f, Rmse %f, delta1 %f, delta2 %f delta3 %f'%(i,len(test_dataset),np.average(errors), np.median(errors), np.sqrt(np.sum(errors * errors)/errors.shape),\
			np.sum(errors < 11.25) / errors.shape[0],np.sum(errors < 22.5) / errors.shape[0],np.sum(errors < 30) / errors.shape[0]))
	del images_tensor, labels_tensor, outputs, masks_tensor

for i, sample_batched in enumerate(test_dataloader):
	train_one_iter(i, sample_batched, 2)
