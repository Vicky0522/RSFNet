import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time

from .palette import image_cluster


class PaletteGenerate(object):
    def __init__(self, gpu_id=0, k=5):
        self.gpu_id = gpu_id
        self.k = k

    def means_to_palette(self, means):
        means = means.repeat(1,1,64,64)
        palette = torch.cat(torch.split(means,1,0), dim=3)
        return palette

    def forward(self, x):
        means = image_cluster(x) 
        means = torch.from_numpy(means).unsqueeze(2).unsqueeze(3).cuda(self.gpu_id).type(torch.float32)/255.
        palette_img = self.means_to_palette(means)

        distance = []
        for i in range(self.k):
            distance.append(torch.sqrt(torch.sum((x - means[i:i+1,...])**2, dim=1, keepdim=True)))
        distance = torch.cat(distance, dim=1)
        mask_distance, mask_label = torch.min(distance, dim=1, keepdim=True)
        mask = torch.zeros((1,self.k,x.size(2),x.size(3)), dtype=torch.float32).to(x.device)
        for i in range(self.k):
            max_dist = torch.max(mask_distance[mask_label==i])
            mask[:,i:i+1,...] = torch.exp(-torch.sum(torch.pow(x - means[i:i+1,...],2),dim=1)/(2*torch.pow(max_dist/2.,2)))
        return mask, palette_img


