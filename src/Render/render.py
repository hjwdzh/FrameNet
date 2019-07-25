from ctypes import *

import numpy as np
Render = cdll.LoadLibrary('./Render/libRender.so')

def setup(info):
	Render.InitializeCamera(info['depthWidth'], info['depthHeight'],
		c_float(info['d_fx']), c_float(info['d_fy']), c_float(info['d_cx']), c_float(info['d_cy']))

def SetMesh(V, F):
	handle = Render.SetMesh(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), V.shape[0], F.shape[0])
	return handle

def render(handle, world2cam):
	Render.SetTransform(handle, c_void_p(world2cam.ctypes.data))
	Render.Render(handle);

def getDepth(info):
	depth = np.zeros((info['depthHeight'],info['depthWidth']), dtype='float32')
	Render.GetDepth(c_void_p(depth.ctypes.data))

	return depth

def getVMap(handle, info):
	vindices = np.zeros((info['depthHeight'],info['depthWidth'], 3), dtype='int32')
	vweights = np.zeros((info['depthHeight'],info['depthWidth'], 3), dtype='float32')

	Render.GetVMap(handle, c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data))

	return vindices, vweights

def colorize(VC, vindices, vweights, mask, cimage):
	Render.Colorize(c_void_p(VC.ctypes.data), c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data),
		c_void_p(mask.ctypes.data), c_void_p(cimage.ctypes.data), vindices.shape[0], vindices.shape[1])

def directionalize(Qx, Qy, ambiguity, vindices, vweights, mask, Q_cam, N_cam, rot, depth, fx, fy, cx, cy):
	print(vindices.shape)
	Render.Directionalize(c_void_p(Qx.ctypes.data), c_void_p(Qy.ctypes.data), c_void_p(ambiguity.ctypes.data), c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data),
		c_void_p(mask.ctypes.data), c_void_p(Q_cam.ctypes.data), c_void_p(N_cam.ctypes.data), c_void_p(rot.ctypes.data), c_void_p(depth.ctypes.data), vindices.shape[0], vindices.shape[1], c_float(fx), c_float(fy), c_float(cx), c_float(cy))

def Clear():
	Render.ClearData()

def visualizeDirection(file, gt_color, Qx, Qy):
	Render.VisualizeDirection(c_char_p(file.encode('utf-8')), c_void_p(gt_color.ctypes.data), c_void_p(Qx.ctypes.data), c_void_p(Qy.ctypes.data), gt_color.shape[0], gt_color.shape[1])

def Rotate(tar, src, rot):
	Render.Rotate(c_void_p(tar.ctypes.data), c_void_p(src.ctypes.data), src.shape[0], c_void_p(rot.ctypes.data))
