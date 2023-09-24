from PIL import Image, ImageCms
import itertools
import numpy as np
import random
import math

import torch
import torch.nn.functional as F

from .util import rgb2lab, distance

def sample_bins(img_pixel_cnt, bin_cnt=16):
	bin_range = 256 // bin_cnt # the pixel range each bin contain

	tmp = {}
	for x in itertools.product(range(bin_cnt),repeat=3):
		tmp[x] = {'val': np.array([0,0,0]), 'cnt': 0}
	for pixel, cnt in img_pixel_cnt.items():
		idx = tuple([c // bin_range for c in pixel])
		tmp[idx]['val'] += np.array(pixel) * cnt
		tmp[idx]['cnt'] += cnt

	res = {}
	for bin_item in tmp.values():
		if bin_item['cnt'] != 0:
			res[tuple((bin_item['val'] / bin_item['cnt']))] = bin_item['cnt']

	return res

def init_means(bins, k=5):

	def attenuation(color,last_mean):
		return 1 - math.exp(((distance(color, last_mean) / 80) ** 2) * -1)

	res = []
	bins = {k: v for k, v in sorted(bins.items(), key=lambda item: item[1], reverse=True)}
	# for color, cnt in bins.items():
	for _ in range(k):
		for color,cnt in bins.items():
			if color not in res: 
				res.append(color)
				break
		bins = {k: v * attenuation(k,res[-1]) for k, v in bins.items()}
		bins = {k: v for k, v in sorted(bins.items(), key=lambda item: item[1], reverse=True)}

	return res

def k_means(bins, k=5, init_mean=True, max_iter=1000, black=True):
	if init_mean is False: means = random.sample(list(bins),k)
	else: means = init_means(bins, k)
	if black: means.append([0, 128, 128])
	means = np.array(means)
	mean_cnt = means.shape[0]

	#cluster_cnt = [0 for i in range(mean_cnt)]
	cluster_cnt = np.zeros(mean_cnt)
	for _ in range(max_iter):
		cluster_sum = [np.array([0,0,0],dtype=float) for i in range(mean_cnt)]
		cluster_cnt = np.zeros(mean_cnt)
		for color, cnt in bins.items():
			color = np.array(color)	
			dists = [distance(color,mean) for mean in means]
			cluster_th = dists.index(min(dists))
			cluster_sum[cluster_th] += color * cnt
			cluster_cnt[cluster_th] += cnt

		new_means = [cluster_sum[i] / cluster_cnt[i] if cluster_cnt[i] > 0 else [0,0,0] for i in range(k)]
		if black: new_means.append([0,128,128])
		new_means = np.array(new_means)
		if (new_means == means).all(): break
		else: means = new_means

	arg_th = np.argsort(means[:k], axis=0)[:,0][::-1]

	return means[arg_th], cluster_cnt[arg_th]

def image_cluster(image, k=5):
    ratio = image.size(3) / image.size(2)
    image_r = F.interpolate(image, size=(512, int(512 * ratio)), mode='bilinear')
    img = Image.fromarray((image_r.squeeze().permute(1,2,0)*255).type(torch.uint8).cpu().numpy())
    colors = img.getcolors(img.size[0] * img.size[1])
    bins = {}
    for count, pixel in colors:
        bins[pixel] = count
    bins = sample_bins(bins)
    means, means_weight = k_means(bins, k=k, init_mean=True)
    
    return means






