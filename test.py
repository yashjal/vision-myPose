import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import pose_estimation
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

boxsize = 368
scale_search = [0.5, 1.0, 1.5, 2.0]
stride = 8
padValue = 0.
thre_point = 0.15
thre_line = 0.05
stickwidth = 4

def construct_model(path):
	model = pose_estimation.PoseModel(num_vertices=19, num_vector=19)
	state_dict = torch.load(path)['state_dict']
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	state_dict = model.state_dict()
	state_dict.update(new_state_dict)
	model.load_state_dict(state_dict)
	model = model.cuda()
	model.eval()
	return model

def padRightDownCorner(img, stride, padValue):
	h = img.shape[0]
	w = img.shape[1]
	pad = 4 * [None]
	pad[0] = 0 # up
	pad[1] = 0 # left
	pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
	pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right
	img_padded = img
	pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
	img_padded = np.concatenate((pad_up, img_padded), axis=0)
	pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
	img_padded = np.concatenate((pad_left, img_padded), axis=1)
	pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
	img_padded = np.concatenate((img_padded, pad_down), axis=0)
	pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
	img_padded = np.concatenate((img_padded, pad_right), axis=1)
	
	return img_padded, pad

def normalize(origin_img):
	origin_img = cv2.resize(origin_img,(368,368))
	origin_img = np.array(origin_img, dtype=np.float32)
	#origin_img = np.transpose(origin_img[:,:,:,np.newaxis], (3, 2, 0, 1))
	origin_img -= 128.0
	origin_img /= 256.0
	#origin_img *= 128.0
	#origin_img += 128.0
	#origin_img /= 256.0
	return origin_img


def process(model, input_path):
	origin_img = cv2.imread(input_path)
	normed_img = normalize(origin_img)
	#with open('heat1/img.pickle', 'wb') as outfile:
		#pickle.dump(normed_img, outfile)
	normed_img = np.transpose(normed_img[:,:,:,np.newaxis], (3, 2, 0, 1))
	# preprocess
	mask = np.ones((1,1,normed_img.shape[2]/stride,normed_img.shape[3]/stride), dtype=np.float32)
	input_var = torch.autograd.Variable(torch.from_numpy(normed_img).cuda())
	mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())
	# get the features
	heat1,vec1,heat2,vec2,heat3,vec3,heat4,vec4 = model(input_var, mask_var)
	# get the heatmap
	heatmap = heat4.data.cpu().numpy()
	heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
	print "HEATMAPS generated"
	heatmap = cv2.resize(heatmap,(368,368))
	for part in range(1, 19):
		print part
		#heatmap = cv2.resize(heatmap,(368,368))
		heat = heatmap[:,:,part]
		heat = heat.reshape((368,368,1))
		heat *= 255
		heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
		# heat = heat.reshape((368,368,1))
		heat /= 255
		with open('heat1/heat'+str(part)+'.pickle', 'wb') as outfile:
			pickle.dump(heat, outfile)
		# result = heat * 0.4 + img * 0.5
		# print part
		# solution to add image on top pickle dump this heat and img with approriate axes
		# print "PLOTTING"
	#plt.imsave('result1.png',heat,format='png')
	#plt.close()
	paf = vec4.data.cpu().numpy()
	paf = np.transpose(np.squeeze(paf), (1, 2, 0))
	print "PAFs generated"
	paf = cv2.resize(paf,(368,368))
	for part in range(0, 38, 2):
		print part
		vec = np.abs(paf[:,:,part])
		vec += np.abs(paf[:,:,part+1])
		vec[vec > 1] = 1
		vec = vec.reshape((368,368,1))
		# vec[vec > 0] = 1
		vec *= 255
		vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
		vec = vec.reshape((368,368))
		vec /= 255
		with open('vec1/vec'+str(part)+'.pickle', 'wb') as outfile:
			pickle.dump(vec, outfile)

if __name__ == '__main__':
	# load model
	model = construct_model('openpose_coco_best.pth.tar')
	# generate image with body parts
	process(model, 'skate.jpg')
