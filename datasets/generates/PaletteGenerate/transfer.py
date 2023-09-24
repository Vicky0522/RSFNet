import math
import itertools
import numpy as np
from util import *
from multiprocessing import Pool, cpu_count
import time
from PIL import Image

def monotonic_luminance_transfer(ori_palette, index, new_l):
	"""
	Args:
		ori_palette: k color palette
		index: index of the changed color palette
		new_l: new set luminance

	Return: 
		modified_palette
	"""
	modified_palette = ori_palette
	modified_palette[index] = (new_l, *modified_palette[index][-2:])
	for i in range(index+1, len(original_p)):
		modified_palette[i] = (min(modified_palette[i][0], modified_palette[i-1][0]), *modified_palette[i][-2:])
	for i in range(index-1, -1, -1):
		modified_palette[i] = (max(modified_palette[i][0], modified_palette[i+1][0]), *modified_palette[i][-2:])

	return modified_palette

def luminance_transfer(pixel_color, ori_palette, modified_palette, weights):
	"""
	Args:
		pixel_color: pixel to be transfered by modified palette
		ori_palette: original palette
		modified_palette: modified palette
		weights: no used
	Return: 
		modified luminance
	"""
	l = pixel_color[0]
	if l <= 0: return 0
	if l > 100: return 100

	ori_l = [100] + [*ori_palette[:,0]] + [0]
	modified_l = [100] + [*modified_palette[:,0]] + [0]

	# find the nearest two palette by luminance
	for i in range(len(ori_l)):
		if ori_l[i] == l: return modified_l[i]
		if ori_l[i] > l > ori_l[i+1]:
			return (modified_l[i+1] * (l - ori_l[i]) + modified_l[i] * (ori_l[i+1] - l)) / (ori_l[i+1] - ori_l[i])

def single_palette_color_transfer(pixel_color, ori_color, modified_color):

	def get_boundary(ori, direct, k_min, k_max, max_iter=20):
		start = ori + k_min * direct
		end = ori + k_max * direct
		for _ in range(max_iter):
			mid = (start + end) / 2
			if ValidLAB(mid) and ValidRGB(LABtoRGB(mid)): start = mid
			else: end = mid

		return (start + end) / 2

	pixel_color = np.array(pixel_color)
	ori_color = np.array(ori_color)
	modified_color = np.array(modified_color)

	offset = modified_color - ori_color

	c_boundary = get_boundary(ori_color, offset, 1,255)
	lab = pixel_color + offset

	if ValidLAB(lab) and ValidRGB(LABtoRGB(lab)):
		x_boundary = get_boundary(pixel_color, offset, 1, 255)
	else:
		x_boundary = get_boundary(modified_color,pixel_color - ori_color,0,1)

	if distance(x_boundary, pixel_color) == 0: return pixel_color
	
	if distance(c_boundary, ori_color) == 0: ratio = 1
	else:
		ratio = min(1, (distance(x_boundary, pixel_color) / distance(c_boundary, ori_color)))
	
	res = pixel_color + ((x_boundary - pixel_color) / distance(x_boundary, pixel_color) * distance(modified_color, ori_color) * ratio)
	return res

def get_weights(pixel_color, ori_palette):
	
	dist = []
	for p1, p2 in itertools.combinations(ori_palette, 2):
		dist.append(distance(p1,p2))
	mean_dist = sum(dist) / len(dist)

	def gaussian(a,b):
		r = distance(a,b)
		return math.exp(-(r**2) / (2 * (mean_dist**2)))
	
	palette_cnt = len(ori_palette)

	p_matrix = np.zeros((palette_cnt,palette_cnt), dtype='float64')
	for i in range(palette_cnt):
		for j in range(palette_cnt):
			p_matrix[i,j] = gaussian(ori_palette[j], ori_palette[i])

	p_matrix = np.linalg.inv(p_matrix)
	lamda = []
	for i in range(palette_cnt):
		ans = np.zeros(palette_cnt)
		ans[i] = 1
		lamda.append(np.dot(ans,p_matrix))

	weights = np.zeros(palette_cnt)
	for i in range(palette_cnt):
		for j in range(palette_cnt):
			weights[i] += lamda[i][j] * gaussian(pixel_color, ori_palette[j])
	weights = [w if w > 0 else 0 for w in weights]
	weights /= np.sum(weights)

	return weights

def rbf_weights(ori_palette, sample_colors):

	sample_weight_map = {}
	ori_palette = np.array([RegularLAB(c) for c in ori_palette], dtype='float64')

	args = []
	for color in sample_colors:
		args.append((RegularLAB(color), ori_palette))

	with Pool(cpu_count() - 1) as pool:
		weights = pool.starmap(get_weights, args)

	for i in range(len(sample_colors)):
		sample_weight_map[sample_colors[i]] = weights[i]

	return sample_weight_map

def multi_palette_color_transfer(pixel_color, ori_palette, modified_palette, weights):
	
	palette_cnt = len(ori_palette)
	store = []
	for i in range(palette_cnt):
		store.append(single_palette_color_transfer(pixel_color, ori_palette[i], modified_palette[i]))

	# weights = rbf_weights(pixel_color, ori_palette)
	res = np.array([0,0,0],dtype=float)

	for w,c in zip(weights, store):
		res += w * c

	return res

def sample_RGB_color(sample_rate=16):

	levels = [round(i * 255/(sample_rate-1), 5) for i in range(sample_rate)]
	colors = []
	for r, g, b in itertools.product(levels, repeat=3):
		colors.append((r,g,b))

	return colors

def find_nearest_corners(target, step, step_range):
	"""
	Args:
		target: RGB pixel color
		step: sample color step size (255 / sample_cnt)
		step_range: sample color step range 
		Ex: divided to three grid (0, 127.5 , 255) step = 127.5, step_range = [0, 127,5 255]
	"""
	corners = []
	for c in target:
		index = c / step
		corners.append((step_range[math.floor(index)], step_range[math.ceil(index)]))

	return corners


def trilinear_interpolation(target, corners, sample_colors_map):

	xyz_dist = []
	for i in range(3):
		dist = (target[i] - corners[i][0]) / (corners[i][1] - corners[i][0]) if corners[i][1] != corners[i][0] else 0
		xyz_dist.append((1-dist, dist))

	# for c in itertools.product(*corners):
		# if c not in sample_colors_map: print(c)

	eight_corners_val = [np.array(sample_colors_map[c]) for c in itertools.product(*corners)]
	x_interp, y_interp, res = [], [], 0
	for i in range(4):
		x_interp.append(eight_corners_val[i] * xyz_dist[0][0] + eight_corners_val[i+4] * xyz_dist[0][1])
	
	for i in range(2):
		y_interp.append(x_interp[i] * xyz_dist[1][0] + x_interp[i+2] * xyz_dist[1][1])
	
	res = y_interp[0] * xyz_dist[2][0] + y_interp[1] * xyz_dist[2][1]

	return res

def img_color_transfer(img, original_p, modified_p, sample_weight_map, sample_colors, sample_rate):

	sample_colors_map = {}
	original_p = np.array([RegularLAB(c) for c in original_p], dtype='float64')
	modified_p = np.array([RegularLAB(c) for c in modified_p], dtype='float64')

	args = []
	for color in sample_colors:
		args.append([RegularLAB(color), original_p, modified_p, sample_weight_map[color]])

	# sample color image transfer
	print('using {} cores CPU'.format(cpu_count() - 1))
	with Pool(cpu_count()) as pool:
		l = pool.starmap(luminance_transfer, args)
		color = pool.starmap(multi_palette_color_transfer, args)
	for i in range(len(sample_colors)):
		sample_colors_map[sample_colors[i]] = ByteLAB((l[i], *color[i][-2:]))

	# image pixel color transfer by sample color trilinear interpolation
	step = 255 / (sample_rate - 1)
	step_range = [round(i * (255/(sample_rate-1)), 5) for i in range(sample_rate)]

	color_map = {}
	colors = img.getcolors(img.width * img.height)

	args = []
	for _,c in colors:
		nearest_corners = find_nearest_corners(c, step, step_range)
		args.append((c, nearest_corners, sample_colors_map))
	with Pool(cpu_count() - 1) as pool:
		interp_res = pool.starmap(trilinear_interpolation, args)
	for i in range(len(colors)):
		color_map[colors[i][1]] = tuple([int(x) for x in interp_res[i]])

	result = Image.new('LAB', img.size)
	result_pixels = result.load()
	img_pixels = img.load()
	for i in range(img.width):
		for j in range(img.height):
			result_pixels[i,j] = color_map[img_pixels[i,j]]

	return lab2rgb(result)

# def single_transfer(pixel_color, original_p, modified_p):
# 	pixel_color = RegularLAB(pixel_color)
# 	weights = get_weights(pixel_color, original_p)
# 	l = luminance_transfer(pixel_color, original_p, modified_p, weights)
# 	res = multi_palette_color_transfer(pixel_color, original_p, modified_p, weights)
# 	return ByteLAB((l, *res[-2:]))

# def test_my_transfer(img, original_p, modified_p):
# 	img_pixels = img.load()
# 	original_p = np.array([RegularLAB(c) for c in original_p])
# 	modified_p = np.array([RegularLAB(c) for c in modified_p])
	
# 	result = Image.new('LAB', img.size)
# 	result_pixels = result.load()

# 	args = []

# 	colors = img.getcolors(img.width * img.height)
# 	colors_map = {}

# 	for _,c in colors: 
# 		args.append((c, original_p, modified_p))

# 	with Pool(cpu_count() - 1) as pool:
# 		color = pool.starmap(single_transfer, args)

# 	for idx,c in enumerate(colors):
# 		colors_map[c[1]] = color[idx]

# 	for i in range(img.width):
# 		for j in range(img.height):
# 			res = colors_map[img_pixels[i,j]]
# 			result_pixels[i,j] = tuple([int(x) for x in res])

# 	result = lab2rgb(result)
# 	result.save('single_result.jpg')
# 	lab2rgb(img).save('single_oringial.jpg')
# 	return result
