import cv2
import numpy as np
from direction import *

color_image = None
original_image = None
update = False
mouse_x = 0
mouse_y = 0
patch = None
dirX = None
dirY = None
intrinsics = None
patch_w = 0.1
patch_c = 0
deformable = 0
mesh_info = None
rot = 0
scale = 5
def UpdateImage():
	global draw_image, mouse_x, mouse_y, intrinsics, dirX, dirY, patch_w, patch_c, color_image, patch, deformable

	draw_image = color_image.copy()
	if deformable == 1:
		outputs = ComputeWarping(dirX, dirY, intrinsics, mouse_x,mouse_y,patch_w, patch_c).astype('float32')
		Render(patch, outputs, draw_image, 0)
		cv2.imshow("TextureAttach", draw_image)
	else:

		H = BuildHomography(intrinsics, np.array([mouse_x, mouse_y]), dirX[mouse_y,mouse_x], dirY[mouse_y,mouse_x], patch_w, patch_c)
		warp = cv2.warpPerspective(patch, H, (color_image.shape[1], color_image.shape[0]))

		mask = (warp[:,:,0] < 10) * (warp[:,:,1]<10) * (warp[:,:,2] < 10)
		mask = np.reshape(mask, (mask.shape[0],mask.shape[1],1))
		mask = np.tile(mask, (1,1,3))
		draw_image = (draw_image * mask + draw_image * (1 - mask) * 0.5 + warp*(1-mask) * 0.5).astype('uint8')
		cv2.imshow("TextureAttach", draw_image)

def ModifyImage():
	global mouse_x, mouse_y, intrinsics, dirX, dirY, patch_w, patch_c, color_image, deformable, mesh_info, rot, scale

	if deformable == 2:
		tangent1 = dirX[mouse_y, mouse_x]
		tangent2 = dirY[mouse_y, mouse_x]

		for j in range(rot):
			temp = tangent1.copy()
			tangent1 = -tangent2.copy()
			tangent2 = temp
		normal = np.cross(tangent1, tangent2)
		x = tangent1
		y = normal
		z = np.cross(x, y)
		rotation = np.array([x, y, z])
		rotation = np.transpose(rotation)
		translation = np.array([(mouse_x - intrinsics[1]) / intrinsics[0], (mouse_y - intrinsics[3]) / intrinsics[2], 1]) * scale
		color,zbuffer = Render3D(mesh_info, intrinsics, rotation, translation)
		mask = np.tile(np.reshape(zbuffer > 0,(480,640,1)),(1,1,3))

		color_image = (color_image * (1 - mask) + color)

	elif deformable == 1:
		outputs = ComputeWarping(dirX, dirY, intrinsics, mouse_x,mouse_y,patch_w, patch_c).astype('float32')
		Render(patch, outputs, color_image, 1)
	else:
		H = BuildHomography(intrinsics, np.array([mouse_x, mouse_y]), dirX[mouse_y,mouse_x], dirY[mouse_y,mouse_x], patch_w, patch_c)
		warp = cv2.warpPerspective(patch, H, (color_image.shape[1], color_image.shape[0]))
		mask = (patch[:,:,0]>= 1) + (patch[:,:,1]>=1) + (patch[:,:,2] >= 1)
		mask = np.reshape(mask > 0, (mask.shape[0],mask.shape[1],1))
		mask = np.tile(mask, (1,1,3)).astype('uint8') * 255
		warp_mask = cv2.warpPerspective(mask, H, (color_image.shape[1], color_image.shape[0]))

		mask = warp_mask == 255
		#mask = np.reshape(mask, (mask.shape[0],mask.shape[1],1))
		#mask = np.tile(mask, (1,1,3))
		color_image = (color_image * (1-mask) + warp * mask).astype('uint8')#color_image * (warp > 0) * 0.5 + warp * 0.5).astype('uint8')

def click_and_crop(event, x, y, flags, param):
	global mouse_x, mouse_y, original_image, color_image
	mouse_x = x
	mouse_y = y
	if event == cv2.EVENT_LBUTTONDOWN:
		ModifyImage()
 
	elif event == cv2.EVENT_MBUTTONDOWN:
		color_image = original_image.copy()

	UpdateImage()

def app(cimage, dirX_3d, dirY_3d, attached_patch, intrinsic, mesh_infos):
	global color_image, original_image, patch_w, update, mouse_y, mouse_x, patch, intrinsics, dirX, dirY, patch_c, deformable, mesh_info, rot, scale
	mesh_info = mesh_infos
	patch_w = 0.2 / attached_patch.shape[0]
	color_image = cimage.copy()
	patch = attached_patch.copy()
	intrinsics = intrinsic.copy()
	dirX = dirX_3d.copy()
	dirY = dirY_3d.copy()

	original_image = color_image.copy()
	update = False
	mouse_x = 0
	mouse_y = 0
	patch_c = patch.shape[0]//2
	cv2.namedWindow("TextureAttach")
	cv2.setMouseCallback("TextureAttach", click_and_crop)
	draw_image = color_image.copy()
	cv2.imshow("TextureAttach", draw_image)

	while True:
		# display the image and wait for a keypress
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'c' key is pressed, break from the loop
		if key == ord("g"):
			break

		if key == ord('d'):
			if deformable < 2:
				patch_w *= 1.1
			else:
				scale *= 1.1
			UpdateImage()

		if key == ord('f'):
			if deformable < 2:
				patch_w /= 1.1
			else:
				scale /= 1.1
			UpdateImage()

		if key == ord('a'):
			deformable = deformable + 1
			if deformable == 3:
				deformable = 0

		if key == ord('s'):
			cv2.imwrite('result.png', color_image)

		if key == ord('r'):
			rot = rot + 1
			if rot == 4:
				rot = 0
			M = cv2.getRotationMatrix2D((patch.shape[1]/2, patch.shape[0]/2), 90, 1)
			patch = cv2.warpAffine(patch, M, (patch.shape[0], patch.shape[1]))
			UpdateImage()