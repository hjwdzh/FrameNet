import numpy as np
from ctypes import *
import cv2
import scipy.misc as misc
#dirlib = cdll.LoadLibrary('./cpp/build/libDirection.dylib')
dirlib = cdll.LoadLibrary('./cpp/build/libDirection.so')

def Color2Vec(image):
	image = image / 255.0 * 2.0 - 1.0
	norm = np.linalg.norm(image, axis=2) + 1e-7
	for j in range(3):
		image[:,:,j] /= norm
	return image

def ImageCoord(height, width, intrinsics):
	xx, yy = np.meshgrid(np.array([i for i in range(width)]), np.array([i for i in range(height)]))
	mesh_x = (xx - intrinsics[1]) / intrinsics[0]
	mesh_y = (yy - intrinsics[3]) / intrinsics[2]
	return mesh_x, mesh_y

def ProjectDir(dir3D, pixel_x, pixel_y):
	dir2D = np.zeros((dir3D.shape[0], dir3D.shape[1], 2))
	dir2D[:,:,0] = dir3D[:,:,0] - pixel_x * dir3D[:,:,2]
	dir2D[:,:,1] = dir3D[:,:,1] - pixel_y * dir3D[:,:,2]
	return dir2D

def VisualizeDirection(file, gt_color, Qx, Qy):
	if gt_color.dtype == 'uint8':
		color = gt_color.astype('float32') / 255.0
	else:
		color = gt_color
	Qx_float = np.ascontiguousarray(Qx.astype('float32'))
	Qy_float = np.ascontiguousarray(Qy.astype('float32'))
	color = np.ascontiguousarray(color.astype('float32'))
	dirlib.VisualizeDirection(c_char_p(file.encode('utf-8')), c_void_p(color.ctypes.data), c_void_p(Qx_float.ctypes.data), c_void_p(Qy_float.ctypes.data), gt_color.shape[0], gt_color.shape[1])

def ComputeWarping(Qx, Qy, intrinsics, px, py, pixel_w, patch_w):
	output = np.zeros((patch_w * 2 + 1, patch_w * 2 + 1, 2), dtype='float32')
	Qx_float = np.ascontiguousarray(Qx.astype('float32'))
	Qy_float = np.ascontiguousarray(Qy.astype('float32'))
	intrinsics_float = np.ascontiguousarray(intrinsics.astype('float32'))

	integer_params = np.array([Qx.shape[0], Qx.shape[1], px, py, patch_w], dtype='int32')

	#framesX = np.zeros((257,257,3), dtype='float32')
	#framesY = np.zeros((257,257,3), dtype='float32')
	dirlib.ComputeWarping(
		c_void_p(integer_params.ctypes.data), c_float(pixel_w),\
		c_void_p(intrinsics_float.ctypes.data),\
		c_void_p(Qx_float.ctypes.data), c_void_p(Qy_float.ctypes.data),\
		c_void_p(output.ctypes.data))
		#c_void_p(framesX.ctypes.data), c_void_p(framesY.ctypes.data))


	#cv2.imwrite('frame1.png', ((framesX + 1) / 2.0 * 255).astype('uint8'))
	#cv2.imwrite('frame2.png', ((framesY + 1) / 2.0 * 255).astype('uint8'))

	return output

def Render(patch, coord, image, solid):
	dirlib.Rasterize(c_void_p(patch.ctypes.data), c_void_p(coord.ctypes.data), c_int(coord.shape[0]), c_void_p(image.ctypes.data), c_int(image.shape[0]), c_int(image.shape[1]), c_int(solid))
	return image

def DrawTriangle(v1,v2,v3,t1,t2,t3,n1,n2,n3,tex, color_image, z_image, intrinsics):
	dirlib.DrawTriangle(
		c_void_p(v1.ctypes.data),c_void_p(v2.ctypes.data),c_void_p(v3.ctypes.data),\
		c_void_p(t1.ctypes.data),c_void_p(t2.ctypes.data),c_void_p(t3.ctypes.data),\
		c_void_p(n1.ctypes.data),c_void_p(n2.ctypes.data),c_void_p(n3.ctypes.data),\
		c_void_p(tex.ctypes.data),c_void_p(color_image.ctypes.data),c_void_p(z_image.ctypes.data),\
		c_void_p(intrinsics.ctypes.data),\
		c_int(tex.shape[1]), c_int(tex.shape[0]), c_int(color_image.shape[1]), c_int(color_image.shape[0]))

def CanonicalPixel(intrinsics, pixel):
	#fx, cx, fy, cy
	return np.array([(pixel[0] - intrinsics[1]) / intrinsics[0], (pixel[1] - intrinsics[3]) / intrinsics[2], 1])

def IntrinsicMatrix(intrinsics):
	m = np.zeros((3,3))
	m[0,0] = intrinsics[0]
	m[0,2] = intrinsics[1]
	m[1,1] = intrinsics[2]
	m[1,2] = intrinsics[3]
	m[2,2] = 1
	return m

def BuildHomography(intrinsics, pixel, dirX, dirY, pixel_w, image_c):
	p_star = CanonicalPixel(intrinsics, pixel)
	homography = np.zeros((3,3))
	homography[:,0] = dirX * pixel_w
	homography[:,1] = dirY * pixel_w
	homography[:,2] = p_star - image_c * (dirX + dirY) * pixel_w
	intrinsic_matrix = IntrinsicMatrix(intrinsics)
	homography = np.dot(intrinsic_matrix, homography)
	patch_coord = np.array([1,0,1])
	#print(np.dot(homography, patch_coord))
	'''
	scale = float(patch_w) / image_w
	homography[0,:] *= scale
	homography[1,:] *= scale
	'''
	return homography

def ProcessOBJ(obj_file,texture_file=''):
	lines = [l.strip() for l in open(obj_file) if l.strip() != '']
	vertices = []
	normals = []
	texs = []
	face_mats = []
	face_Tinds = []
	face_Ninds = []
	face_Vinds = []
	mat_type = -1
	Imgs = []
	for l in lines:
		words = [w for w in l.split(' ') if w != '']
		if words[0] == 'v':
			vertices.append([float(words[1]), float(words[2]), float(words[3])])
		if words[0] == 'vt':
			texs.append([float(words[1]), float(words[2])])
		if words[0] == 'vn':
			normals.append([float(words[1]), float(words[2]), float(words[3])])
		if words[0] == 'usemtl':
			if (texture_file == ''):
				if words[1] == 'blinn1SG':
					mat_type = len(Imgs)
					img = np.zeros((2,2,3),dtype='uint8')
					img[:,:,0] = 0.59 * 255
					img[:,:,1] = 0.63 * 255
					img[:,:,2] = 0.66 * 255
					Imgs.append(img.copy())
				elif words[1] == 'lambert2SG':
					img = cv2.imread('resources/Converse_obj/converse.jpg')
					img = misc.imresize(img, (256,256))
					mat_type = len(Imgs)
					Imgs.append(img.copy())
				elif words[1] == 'lambert3SG':
					img = cv2.imread('resources/Converse_obj/laces.jpg')
					img = misc.imresize(img, (256,256))
					mat_type = len(Imgs)
					Imgs.append(img.copy())
				else:
					print('wrong!')
					exit(0)
			else:
				img = cv2.imread(texture_file)
				mat_type = len(Imgs)
				Imgs.append(img.copy())
		if words[0] == 'f':
			if mat_type == -1:
				print('wrong')
				exit(0)
			vinds = []
			tinds = []
			ninds = []
			for j in range(3):
				ws = words[j + 1].split('/')
				vinds.append(int(ws[0]))
				tinds.append(int(ws[1]))
				ninds.append(int(ws[2]))
			face_Vinds.append(vinds)
			face_Tinds.append(tinds)
			face_Ninds.append(ninds)
			face_mats.append(mat_type)
			if len(words) == 5:
				vinds = []
				tinds = []
				for j in range(3):
					p = j + 2
					if j == 0:
						p = 1
					ws = words[p].split('/')
					vinds.append(int(ws[0]))
					tinds.append(int(ws[1]))
					ninds.append(int(ws[2]))
				face_Vinds.append(vinds)
				face_Tinds.append(tinds)
				face_Ninds.append(ninds)
				face_mats.append(mat_type)


	vertices = np.array(vertices, dtype='float32')
	texs = np.array(texs, dtype='float32')
	face_Vinds = np.array(face_Vinds, dtype='int32') - 1
	face_Tinds = np.array(face_Tinds, dtype='int32') - 1
	face_Ninds = np.array(face_Ninds, dtype='int32') - 1
	face_mats = np.array(face_mats, dtype='int32')

	vertices[:,1] = -vertices[:,1]
	vertices[:,2] = -vertices[:,2]
	min_v = np.array([np.min(vertices[:,i]) for i in range(3)])
	max_v = np.array([np.max(vertices[:,i]) for i in range(3)])
	max_len = np.max(max_v - min_v)
	vertices /= max_len
	vertices = np.ascontiguousarray(vertices.astype('float32'))

	return {'v':vertices, 't':texs, 'n':normals, 'fv':face_Vinds, 'ft':face_Tinds, 'fn':face_Ninds, 'm':face_mats, 'tex':Imgs}


def Render3D(mesh_info, intrinsics, rotation, translation):
	vertices= mesh_info['v']
	vertices = np.dot(vertices, np.transpose(rotation))
	for j in range(3):
		vertices[:,j] += translation[j]
	vertices = np.ascontiguousarray(vertices.astype('float32'))

	normals = mesh_info['n']
	normals = np.dot(normals, rotation)
	normals = np.ascontiguousarray(normals.astype('float32'))
	texs = mesh_info['t']
	face_Vinds = mesh_info['fv']
	face_Tinds = mesh_info['ft']
	face_Ninds = mesh_info['fn']
	face_mats = mesh_info['m']
	textures = mesh_info['tex']
	color_image = np.zeros((480,640,3),dtype='uint8')
	z_image = np.zeros((480,640),dtype='float32')
	for i in range(0,face_Vinds.shape[0]):
		v1 = vertices[face_Vinds[i][0]]
		v2 = vertices[face_Vinds[i][1]]
		v3 = vertices[face_Vinds[i][2]]
		t1 = texs[face_Tinds[i][0]]
		t2 = texs[face_Tinds[i][1]]
		t3 = texs[face_Tinds[i][2]]
		n1 = normals[face_Ninds[i][0]]
		n2 = normals[face_Ninds[i][1]]
		n3 = normals[face_Ninds[i][2]]
		mat_id = face_mats[i]
		texture = textures[mat_id]
		DrawTriangle(v1,v2,v3,t1,t2,t3,n1,n2,n3,texture, color_image, z_image, intrinsics)
	return color_image, z_image