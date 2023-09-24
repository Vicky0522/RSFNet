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

from .networks.poolnet import build_model, weights_init


class SaliencyDetect(object):
    def __init__(self, gpu_id=0, arch='resnet', weight=""):
        self.arch = arch
        self.gpu_id = gpu_id
        self.weight = weight

        self.net = build_model(self.arch)
        self.net = self.net.to("cuda:{}".format(self.gpu_id))
        
        self.net.load_state_dict(torch.load(self.weight))
        self.net.eval()

        # normalize data
        self.mean = torch.Tensor([104.00699, 116.66877, 122.67892]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to("cuda:{}".format(self.gpu_id))   

    def forward(self, x):
        out = torch.flip(x, [1,]) * 255.
        out = out - self.mean
        preds = self.net(out)
        preds = torch.sigmoid(preds)
        return preds


