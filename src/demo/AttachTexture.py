import cv2
import skimage.io as sio
import sys
import numpy as np
from visualizer import app
from direction import *
import scipy.misc as misc
import argparse

parser = argparse.ArgumentParser(description='Process saome integers.')
parser.add_argument('--input', type=str, default='selected/0000')
parser.add_argument('--resource', type=str, default='resources/im4.png')
args = parser.parse_args()
# set up file names
start_name = args.input
color_name = start_name + '-color.png'
orient_x_name = start_name + '-orient-X_pred.png'
orient_y_name = start_name + '-orient-Y_pred.png'

intrinsics = np.array([577.591,318.905,578.73,242.684]).astype('float32')

if args.resource[-3:] == 'obj':
	mesh_info = ProcessOBJ(args.resource,args.resource[:-3] + 'jpg')
else:
	mesh_info = ProcessOBJ('resources/objects/toymonkey/toymonkey.obj','resources/objects/toymonkey/toymonkey.jpg')

color_image = cv2.imread(color_name)
dirX_3d = Color2Vec(misc.imresize(sio.imread(orient_x_name), (480,640), 'nearest'))
dirY_3d = Color2Vec(misc.imresize(sio.imread(orient_y_name), (480,640), 'nearest'))

if args.resource[-3:] != 'obj':
	resource_image = cv2.imread(args.resource)
else:
	resource_image = cv2.imread('resources/im4.png')

if (resource_image.shape[2] == 4):
	mask = resource_image[:,:,3] > 0
	resource_image[:,:,0] *= mask
	resource_image[:,:,1] *= mask
	resource_image[:,:,2] *= mask
	resource_image = resource_image[:,:,0:3]
min_v = np.min([resource_image.shape[0], resource_image.shape[1]])
start_x = (resource_image.shape[1] - min_v)//2
start_y = (resource_image.shape[0] - min_v)//2
attached_patch = np.ascontiguousarray(resource_image[start_y:start_y+min_v, start_x:start_x+min_v,:])
attached_patch = misc.imresize(attached_patch, (401,401),'nearest')

app(color_image, dirX_3d, dirY_3d, attached_patch, intrinsics, mesh_info)